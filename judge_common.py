#!/usr/bin/env python3
"""
Shared ground for the two judge scripts: the amp-evaluation contract.

`jev_judge.py` and `llm_judge.py` answer the same question (how good was this
agent response?) against the same rubrics, the same sample data and the same
result shape. Everything they share lives here so that a difference in their
scores is a difference between Jev and an LLM, not between two scripts.

From wso2/agent-manager's `libs/amp-evaluation` this keeps:
  - the input:   a Trace with `input` and `output`, plus an optional Task with
                 `success_criteria`, loaded from that library's own
                 `samples/data/sample_traces.json` and `sample_dataset.json`
  - the rubrics: the evaluation steps and the five 0.0/0.25/0.5/0.75/1.0 levels,
                 copied verbatim from `evaluators/builtin/llm_judge.py`
  - the output:  EvalResult(score 0.0-1.0, passed = score >= 0.5, explanation)

It drops the framework: no Param descriptors, no level/mode autodetection, no
aggregators, no runner. These are scripts, not a port.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# amp-evaluation's data model, reduced to what a trace-level judge reads.
# ---------------------------------------------------------------------------

# Unmodified copies of amp-evaluation's own sample data, vendored under data/
# so a clone runs offline and every run scores the same bytes. See NOTICE.
DATA_DIR = Path(__file__).parent / "data"
SAMPLE_TRACES = DATA_DIR / "sample_traces.json"
SAMPLE_DATASET = DATA_DIR / "sample_dataset.json"

@dataclass
class Trace:
    """The trace-level slice of amp_evaluation.trace.models.Trace."""

    trace_id: str
    input: str
    output: str
    start_time: Optional[str] = None


@dataclass
class Task:
    """The dataset-task slice of amp_evaluation.dataset.models.Task."""

    task_id: str
    input: str
    success_criteria: Sequence[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """amp_evaluation.models.EvalResult, same contract, no Param plumbing.

    score is 0.0-1.0 with 0.0 the worst outcome; passed defaults to score >= 0.5;
    skip() means the evaluation could not be performed at all.
    """

    score: Optional[float] = None
    passed: Optional[bool] = None
    explanation: Optional[str] = None
    skip_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.score is None:
            return
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")
        if self.passed is None:
            self.passed = self.score >= 0.5

    @property
    def is_skipped(self) -> bool:
        return self.skip_reason is not None

    @classmethod
    def skip(cls, reason: str) -> "EvalResult":
        return cls(skip_reason=reason)


# ---------------------------------------------------------------------------
# The rubrics. instructions + rubric are lifted from amp-evaluation's
# builtin/llm_judge.py build_prompt() methods. The only edit is the framing
# sentence that inlines the trace text: llm_judge.py puts that text back into
# the prompt, jev_judge.py passes it as state, so it is not stored here.
# ---------------------------------------------------------------------------


@dataclass
class JudgeSpec:
    """One amp LLM-judge evaluator: what to assess, and the five levels."""

    name: str
    description: str
    instructions: str
    rubric: Sequence[str]
    uses_success_criteria: bool = False

    def instructions_for(self, task: Optional[Task] = None) -> str:
        """amp appends the dataset's success_criteria when running an experiment."""
        if self.uses_success_criteria and task and task.success_criteria:
            extra = "; ".join(task.success_criteria)
            return f"{self.instructions}\n\nAdditional success criteria: {extra}"
        return self.instructions


HELPFULNESS = JudgeSpec(
    name="helpfulness",
    description=(
        "Scores whether the response actually helps the user with what they asked for. "
        "Checks for actionable, useful content vs empty acknowledgments."
    ),
    instructions=(
        "You are an expert evaluator. Your sole criterion is HELPFULNESS: does "
        "`agent_response` actually help the user with what they asked for in `user_query`?\n\n"
        "Evaluation Steps:\n"
        "1. Identify what the user needs: what problem are they trying to solve or what "
        "information are they seeking?\n"
        "2. Assess whether the response provides actionable, useful content that moves the "
        "user closer to their goal.\n"
        "3. Check for empty helpfulness: does the response acknowledge the question without "
        'actually helping (e.g. "That\'s a great question! There are many factors to '
        'consider..." without providing the factors)?\n'
        "4. Assess whether the response would leave the user better off than before they asked."
    ),
    rubric=[
        "Not helpful at all; ignores the user's need, provides nothing useful, or answers a "
        "completely different question",
        "Minimally helpful; touches on the topic but does not provide enough useful content "
        "to meaningfully assist the user",
        "Somewhat helpful; provides some useful content but the user would still need "
        "significant additional help",
        "Helpful; addresses the user's need well with only minor gaps in usefulness",
        "Highly helpful; directly and fully assists the user with clear, actionable, and "
        "complete content",
    ],
    uses_success_criteria=True,
)

CLARITY = JudgeSpec(
    name="clarity",
    description=(
        "Scores the response for readability, structure, and absence of ambiguity. "
        "Checks whether the detail level matches the user's apparent expertise."
    ),
    instructions=(
        "You are an expert evaluator. Your sole criterion is CLARITY: is `agent_response` "
        "clear, well-structured, and easy to understand?\n\n"
        "Evaluation Steps:\n"
        "1. Assess readability: can the response be understood on first reading without "
        "re-reading or guessing at meaning?\n"
        "2. Check structure: is the information organized logically? Are related points "
        "grouped together? Does it use formatting (lists, paragraphs) appropriately?\n"
        "3. Check for ambiguity: are there statements that could be interpreted multiple "
        "ways, or vague language where precision is needed?\n"
        "4. Assess whether the level of technical detail matches what `user_query` suggests "
        "about the user's expertise."
    ),
    rubric=[
        "Incomprehensible; disorganized, ambiguous, or impossible to follow",
        "Difficult to understand; poor structure, significant ambiguity, or explanation that "
        "confuses more than it clarifies",
        "Understandable with effort; some structural issues or unclear passages but the core "
        "message comes through",
        "Clear and well-structured; easy to follow with only minor areas that could be clearer",
        "Exceptionally clear; well-organized, unambiguous, and perfectly pitched to the "
        "user's level of understanding",
    ],
)

COMPLETENESS = JudgeSpec(
    name="completeness",
    description=(
        "Checks whether the final response addresses all sub-questions and requirements in "
        "the input. Accepts optional success_criteria. 0.0 = nothing addressed, 1.0 = fully "
        "covered."
    ),
    instructions=(
        "You are an expert evaluator. Your sole criterion is COMPLETENESS: does "
        "`agent_response` address every part of `user_query` without leaving gaps?\n\n"
        "Evaluation Steps:\n"
        "1. Break the user's query into its distinct sub-questions or requirements.\n"
        "2. For each sub-question, check whether the response provides a substantive answer.\n"
        "3. Identify any requirements that are ignored, only partially addressed, or left "
        "unresolved.\n"
        "4. Score based on the proportion of requirements that are adequately covered."
    ),
    rubric=[
        "None of the query's requirements are addressed",
        "Only a small fraction of requirements are addressed; most are missing",
        "Roughly half the requirements are addressed; significant gaps remain",
        "Most requirements are addressed; only minor points are missing",
        "Every requirement and sub-question is fully and substantively covered",
    ],
    uses_success_criteria=True,
)

JUDGES: Dict[str, JudgeSpec] = {j.name: j for j in (HELPFULNESS, CLARITY, COMPLETENESS)}

# The numeric labels amp prints beside each rubric level.
RUBRIC_LABELS = ("0.0 ", "0.25", "0.5 ", "0.75", "1.0 ")


# ---------------------------------------------------------------------------
# The trace text both judges read
# ---------------------------------------------------------------------------


def build_state(trace: Trace) -> Dict[str, str]:
    """The two fields every judge is shown, whole and untruncated."""
    return {"user_query": trace.input, "agent_response": trace.output}


def skip_all(judges: Sequence[JudgeSpec], reason: str) -> Dict[str, EvalResult]:
    return {j.name: EvalResult.skip(reason) for j in judges}


# ---------------------------------------------------------------------------
# amp-evaluation's own sample data
# ---------------------------------------------------------------------------


def _workflow_query(raw: Dict[str, Any]) -> str:
    """The user's message, from the agent workflow span rather than the root.

    Only 3 of the 14 sample traces carry input/output at the trace root; the
    rest are rooted at `POST /chat` and keep the conversation on a nested
    `*.workflow` span. amp's own trace parser reads the spans, so this does
    too, which is the difference between 3 scorable traces and 10.
    """
    for span in raw.get("spans", []):
        attrs = span.get("attributes", {})
        entity_input = attrs.get("traceloop.entity.input")
        if not (span.get("name", "").endswith(".workflow") and entity_input):
            continue
        try:
            messages = json.loads(entity_input)["inputs"]["messages"]
        except (ValueError, KeyError, TypeError):
            continue
        if messages:
            last = messages[-1]
            return (last if isinstance(last, str) else str(last.get("content", last))).strip()
    return ""


def _final_answer(raw: Dict[str, Any]) -> str:
    """The last assistant completion in the trace: what the user actually saw."""
    answer = ""
    for span in raw.get("spans", []):
        content = span.get("attributes", {}).get("gen_ai.completion.0.content")
        if content:
            answer = content
    return answer.strip()


def load_traces(path: Optional[Path] = None) -> List[Trace]:
    source = path or SAMPLE_TRACES
    payload = json.loads(source.read_text())
    traces = []
    for raw in payload.get("traces", []):
        query = _workflow_query(raw)
        answer = _final_answer(raw) or (raw.get("output") or "").strip()
        traces.append(
            Trace(
                trace_id=raw["traceId"],
                input=query,
                output=answer,
                start_time=raw.get("startTime"),
            )
        )
    return traces


def load_tasks(path: Optional[Path] = None) -> List[Task]:
    source = path or SAMPLE_DATASET
    payload = json.loads(source.read_text())
    return [
        Task(
            task_id=raw["task_id"],
            input=raw["input"],
            success_criteria=raw.get("success_criteria", []),
        )
        for raw in payload.get("tasks", [])
    ]


# ---------------------------------------------------------------------------
# The five traces the comparison runs on
# ---------------------------------------------------------------------------

# Ten of amp's fourteen sample traces carry a real exchange; these five were
# picked to spread the rubric rather than to flatter either judge. Each is a
# different way an agent can end a turn, and two pairs are deliberately close:
# 789a4cc3 vs 7ec82d87 both follow a tool failure, and 7ec82d87 vs 3c6c9d5f
# both leave the user without a booking for opposite reasons.
INTERESTING_TRACES: Dict[str, str] = {
    "fc5513186f8d0b0d0b488f47548e6028": "substantive answer, but ignores 'next week'",
    "789a4cc3a165ed330d3244aca8b61dbb": "tool failed; generic advice dressed as help",
    "7ec82d8703e40682c0850b9b02aeae03": "tool failed; gives the user nothing",
    "3c6c9d5f6890ef25bdf9694a80be921b": "vague request, correctly asks for details",
    "59ab6f6cea2ca1e11e24ca140e98f2c7": "starts a plan, then stops to ask one question",
}


# ---------------------------------------------------------------------------
# Shared CLI plumbing
# ---------------------------------------------------------------------------


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--traces", type=Path, help="Path to a sample_traces.json (default: fetched)")
    parser.add_argument("--dataset", type=Path, help="Path to a sample_dataset.json (default: fetched)")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N traces")
    parser.add_argument(
        "--judges",
        default=",".join(JUDGES),
        help=f"Comma-separated subset of: {', '.join(JUDGES)}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score every trace that has an exchange, not just the curated five",
    )
    parser.add_argument("--json", action="store_true", help="Emit EvaluatorScore-shaped JSON")
    return parser


def select_judges(parser: argparse.ArgumentParser, spec: str) -> List[JudgeSpec]:
    try:
        judges = [JUDGES[n.strip()] for n in spec.split(",") if n.strip()]
    except KeyError as exc:
        parser.error(f"unknown judge {exc}; choose from {', '.join(JUDGES)}")
    if not judges:
        parser.error("no judges selected")
    return judges


def load_corpus(args: argparse.Namespace) -> List[tuple]:
    """Select the traces to score and pair each with its dataset Task, if any."""
    traces = [t for t in load_traces(args.traces) if t.input and t.output]
    if not getattr(args, "all", False):
        by_id = {t.trace_id: t for t in traces}
        missing = [tid for tid in INTERESTING_TRACES if tid not in by_id]
        if missing:
            print(
                f"warning: {len(missing)} curated trace id(s) not in this data; "
                "falling back to every trace with an exchange",
                file=sys.stderr,
            )
        else:
            traces = [by_id[tid] for tid in INTERESTING_TRACES]
    if args.limit:
        traces = traces[: args.limit]
    tasks = {t.input.strip().lower(): t for t in load_tasks(args.dataset)}
    return [(t, tasks.get(t.input.strip().lower())) for t in traces]


def cell(result: EvalResult) -> str:
    if result.is_skipped:
        return "skip"
    return f"{result.score:.2f} {'P' if result.passed else 'F'}"


def print_table(rows: List[Dict[str, Any]], judges: Sequence[JudgeSpec]) -> None:
    header = f"{'trace':<14}" + "".join(f"{j.name:>16}" for j in judges)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['trace_id'][:12]:<14}"
        for judge in judges:
            line += f"{cell(row['results'][judge.name]):>16}"
        note = INTERESTING_TRACES.get(row["trace_id"])
        print(line + (f"   {note}" if note else ""))


def _record(row: Dict[str, Any], judge: JudgeSpec) -> Dict[str, Any]:
    result = row["results"][judge.name]
    record = {
        "evaluator": judge.name,
        "trace_id": row["trace_id"],
        "trace_start_time": row["trace_start_time"],
        "task_id": row["task_id"],
        "score": result.score,
        "passed": result.passed,
        "explanation": result.explanation,
        "skip_reason": result.skip_reason,
    }
    # Present only when a script ran a second judge over the same trace.
    baseline = row.get("baseline", {}).get(judge.name)
    if baseline is not None:
        record["baseline_score"] = baseline.score
        record["baseline_passed"] = baseline.passed
        record["baseline_explanation"] = baseline.explanation
        record["baseline_skip_reason"] = baseline.skip_reason
    return record


def print_json(rows: List[Dict[str, Any]], judges: Sequence[JudgeSpec]) -> None:
    """amp's EvaluatorScore field names, one record per (trace, evaluator)."""
    print(json.dumps([_record(row, judge) for row in rows for judge in judges], indent=2))
