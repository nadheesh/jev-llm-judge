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
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# amp-evaluation's data model, reduced to what a trace-level judge reads.
# ---------------------------------------------------------------------------

# Unmodified copies of amp-evaluation's own sample data, vendored under data/
# so a clone runs offline and every run scores the same bytes. See NOTICE.
def _load_dotenv() -> None:
    """Read a sibling .env into os.environ, without taking a dependency.

    Both scripts authenticate from environment variables. Loading the file here
    means `python jev_judge.py` works straight after `cp .env.example .env`,
    with no `set -a` dance and no python-dotenv. Real environment variables
    always win, so an export still overrides the file.
    """
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


DATA_DIR = Path(__file__).parent / "data"
SAMPLE_TRACES = DATA_DIR / "sample_traces.json"
SAMPLE_DATASET = DATA_DIR / "sample_dataset.json"

@dataclass
class ToolCall:
    """The content half of amp_evaluation.trace.models.ToolSpan."""

    name: str
    arguments: str = ""
    result: str = ""
    error: str = ""

    def __str__(self) -> str:
        """ToolSpan.__str__, which is what amp puts in an evaluation prompt."""
        if self.error:
            return f"Tool '{self.name}': FAILED ({self.error})"
        return f"Tool '{self.name}': {self.result[:500] if self.result else '(no result)'}"


@dataclass
class Trace:
    """The trace-level slice of amp_evaluation.trace.models.Trace."""

    trace_id: str
    input: str
    output: str
    start_time: Optional[str] = None
    tool_calls: List["ToolCall"] = field(default_factory=list)

    def format_evidence(self) -> str:
        """Trace.format_evidence(): what groundedness checks claims against."""
        if not self.tool_calls:
            return "(no evidence available)"
        lines = "\n".join(f"  {t}" for t in self.tool_calls[:10])
        return f"Tool Results:\n{lines}"

    def format_steps(self) -> str:
        """AgentTrace.format_steps(): the trajectory path_efficiency scores."""
        if not self.tool_calls:
            return "  (no steps recorded)"
        return "\n".join(
            f"  Step {i + 1}: Tool '{t.name}'({t.arguments[:160]})"
            f"{f' FAILED ({t.error})' if t.error else ' -> ok'}"
            for i, t in enumerate(self.tool_calls)
        )

    def has_evidence(self) -> bool:
        return bool(self.tool_calls)


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
# builtin/llm_judge.py build_prompt() methods. All 25 rubric levels and all but
# one evaluation step are verbatim. Two deviations, both forced by the fact that
# Jev takes named state instead of one prompt string:
#
#   * The framing sentence that inlines the trace text is dropped. llm_judge.py
#     puts that text back into the prompt; jev_judge.py passes it as state.
#   * Groundedness step 2 reads "the evidence" where amp reads "the evidence
#     above", since the evidence is a named field rather than text further up.
#
# Both judges here read the same strings, so neither deviation favours one.
# ---------------------------------------------------------------------------


@dataclass
class JudgeSpec:
    """One amp LLM-judge evaluator: what to assess, and the five levels."""

    name: str
    description: str
    instructions: str
    rubric: Sequence[str]
    uses_success_criteria: bool = False
    # What the judge is shown beyond the query and the response. amp encodes
    # this by which trace accessors each build_prompt() reaches for.
    needs_evidence: bool = False
    needs_trajectory: bool = False
    # amp's LLMAsJudgeEvaluator._requires_response_output. Judges that score the
    # trajectory rather than the reply still run when the reply is empty.
    requires_response_output: bool = True
    # amp's GroundednessEvaluator.evaluate() skips when there is nothing to
    # check claims against, rather than scoring a blank zero.
    skip_without_evidence: bool = False

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

GROUNDEDNESS = JudgeSpec(
    name="groundedness",
    description=(
        "Verifies that factual claims in the response are grounded in tool results or "
        "retrieved documents. Skips when no evidence is available."
    ),
    instructions=(
        "You are an expert evaluator. Your sole criterion is GROUNDEDNESS: are the factual "
        "claims in `agent_response` grounded in `evidence`, which is what was actually "
        "available to the agent?\n\n"
        "Evaluation Steps:\n"
        "1. Identify each factual claim in the response (specific facts, numbers, references, "
        "or assertions presented as true).\n"
        "2. For each claim, check whether the evidence directly supports it.\n"
        "3. Classify each claim as: SUPPORTED (evidence backs it), UNSUPPORTED (no relevant "
        "evidence found), or CONTRADICTED (evidence disagrees).\n"
        "4. Score based on the proportion of supported claims. Penalize contradictions more "
        "heavily than unsupported claims.\n\n"
        "Do NOT penalize opinions, hedged statements, or general knowledge that does not need "
        "source evidence. Only assess specific factual claims."
    ),
    rubric=[
        "Most claims are fabricated or contradict the available evidence",
        "Many claims lack support; one or more are contradicted by evidence",
        "Mixed: some claims are supported, others are not; no major contradictions",
        "Most claims are supported by evidence; only minor unsupported details",
        "Every factual claim is grounded in the provided evidence",
    ],
    needs_evidence=True,
    skip_without_evidence=True,
)

PATH_EFFICIENCY = JudgeSpec(
    name="path_efficiency",
    description=(
        "Scores whether the agent's execution path is efficient. "
        "Detects redundant steps, loops, and wasted work."
    ),
    instructions=(
        "You are an expert evaluator. Your sole criterion is PATH EFFICIENCY: does the agent "
        "achieve its goal without unnecessary steps, redundancy, or wasted work? Judge "
        "`execution_steps` against the goal in `user_query`.\n\n"
        "Evaluation Steps:\n"
        "1. Check for redundant steps: is the same tool called with the same or very similar "
        "arguments multiple times? Is the same information retrieved or computed more than "
        "once?\n"
        "2. Check for loops: does the agent repeat the same sequence of actions without making "
        "progress?\n"
        "3. Check for irrelevant steps: are there tool calls or reasoning steps that do not "
        "contribute to the goal at all?\n"
        "4. Assess overall efficiency: could the same result have been achieved with noticeably "
        "fewer steps?"
    ),
    rubric=[
        "Highly inefficient; stuck in loops, significant redundancy, or many irrelevant steps",
        "Several unnecessary steps, repeated actions, or clearly suboptimal tool usage",
        "Moderately efficient; some unnecessary steps but generally making progress toward the "
        "goal",
        "Mostly efficient; at most one or two minor redundancies",
        "Optimally efficient; every step is necessary and no obviously shorter path was "
        "available",
    ],
    needs_trajectory=True,
    requires_response_output=False,
)

JUDGES: Dict[str, JudgeSpec] = {
    j.name: j
    for j in (HELPFULNESS, CLARITY, COMPLETENESS, GROUNDEDNESS, PATH_EFFICIENCY)
}

# The numeric labels amp prints beside each rubric level.
RUBRIC_LABELS = ("0.0 ", "0.25", "0.5 ", "0.75", "1.0 ")


# ---------------------------------------------------------------------------
# The trace text both judges read
# ---------------------------------------------------------------------------


def build_state(trace: Trace, judges: Sequence[JudgeSpec]) -> Dict[str, str]:
    """What this set of judges is shown, whole and untruncated.

    Only the fields the selected judges actually read are included, because in
    amp each build_prompt() decides for itself what to reach for: helpfulness
    never sees the trajectory, and groundedness is the only one given evidence.
    Handing every judge everything would quietly change what they score.
    """
    state = {"user_query": trace.input, "agent_response": trace.output}
    if any(j.needs_evidence for j in judges):
        state["evidence"] = trace.format_evidence()
    if any(j.needs_trajectory for j in judges):
        state["execution_steps"] = trace.format_steps()
        state["total_steps"] = str(len(trace.tool_calls))
    return state


def state_key(judge: JudgeSpec) -> tuple:
    """Judges sharing a key see identical state, so they can share one call."""
    return (judge.needs_evidence, judge.needs_trajectory)


def preflight(trace: Trace, judge: JudgeSpec) -> Optional[EvalResult]:
    """amp's per-evaluator guards, applied before any model is called."""
    if judge.requires_response_output and not trace.output.strip():
        # _requires_response_output: scoring a blank response is misleading.
        return EvalResult.skip("Trace has no response output")
    if judge.skip_without_evidence and not trace.has_evidence():
        # GroundednessEvaluator.evaluate(): nothing to check claims against.
        return EvalResult.skip("No tool or retrieval spans found in this trace")
    return None


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


def _tool_calls(raw: Dict[str, Any]) -> List[ToolCall]:
    """Every tool execution in the trace, in span order.

    Traceloop marks these with span kind "tool". The LangGraph plumbing spans
    around them (tools.task, tools_condition.task, handle_tool_error.task) are
    control flow, not tool calls, and are deliberately left out: counting them
    would inflate the step count that path_efficiency scores.
    """
    calls = []
    for span in raw.get("spans", []):
        attrs = span.get("attributes", {})
        if attrs.get("traceloop.span.kind") != "tool":
            continue
        arguments = attrs.get("traceloop.entity.input", "")
        try:
            arguments = json.loads(arguments).get("input_str", arguments)
        except (ValueError, TypeError, AttributeError):
            pass
        calls.append(
            ToolCall(
                name=attrs.get("traceloop.entity.name") or span.get("name", "").removesuffix(".tool"),
                arguments=str(arguments),
                result=str(attrs.get("traceloop.entity.output", "")),
                error=attrs.get("error.type", "") if span.get("status") == "Error" else "",
            )
        )
    return calls


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
                tool_calls=_tool_calls(raw),
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
    width = max(16, max(len(j.name) for j in judges) + 2)
    header = f"{'trace':<14}" + "".join(f"{j.name:>{width}}" for j in judges)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['trace_id'][:12]:<14}"
        for judge in judges:
            line += f"{cell(row['results'][judge.name]):>{width}}"
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
