#!/usr/bin/env python3
"""
jev_judge, the same amp-evaluation evaluators, scored by Jev instead.

The question: given the inputs and rubrics that wso2/agent-manager's
`libs/amp-evaluation` feeds to an LLM judge, can Jev (TypeSafe AI's System One
model) produce the same scores?

What changes against `llm_judge.py`:
  - An LLM judge inlines the trace into a prompt and parses JSON back. Jev takes
    one `state` and a *map* of questions, so all three judges score a trace in a
    single call instead of three.
  - Each amp rubric is five levels, so it becomes a Score question whose five
    criteria are the five level descriptions. Jev returns a probability-weighted
    level in 0..4; divided by 4 that is amp's 0.0-1.0 score.
  - The explanation is gone. Jev does not generate prose, so the explanation
    field carries the rubric level and Jev's confidence instead of an argument.

Everything else, the input traces, the rubric text and the EvalResult contract,
is shared with `llm_judge.py` through `judge_common.py`.

Usage:
    export JEV_API_KEY=...             # or TYPESAFE_API_KEY
    python jev_judge.py                # score the five curated traces
    python jev_judge.py --all          # every trace with a real exchange
    python jev_judge.py --json         # EvaluatorScore-shaped JSON
    python jev_judge.py --limit 3 --judges helpfulness,clarity
    python jev_judge.py --compare      # Jev and the LLM judge side by side

Requires: pip install typesafe-sdk (plus openai and OPENAI_API_KEY for --compare)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import judge_common as jc
from judge_common import EvalResult, JudgeSpec, Task, Trace

try:
    from typesafe_sdk import Score, TypeSafeClient
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("typesafe-sdk is required: pip install typesafe-sdk")


def build_question(judge: JudgeSpec, task: Optional[Task] = None) -> Score:
    """One amp rubric as a Jev Score question: five levels, five criteria."""
    return Score(instructions=judge.instructions_for(task), criteria=list(judge.rubric))


def to_result(judge: JudgeSpec, answer: Any) -> EvalResult:
    """Map a Jev ScoreAnswer onto amp's 0.0-1.0 EvalResult."""
    levels = len(judge.rubric) - 1
    return EvalResult(
        score=max(0.0, min(1.0, answer.score / levels)),
        explanation=(
            f"jev rubric level {answer.score:.2f}/{levels} "
            f"(confidence {answer.confidence:.2f}) [model=jev]"
        ),
    )


def score_trace(
    client: TypeSafeClient,
    trace: Trace,
    judges: Sequence[JudgeSpec],
    task: Optional[Task] = None,
) -> Dict[str, EvalResult]:
    """One Jev call, every judge scored. Returns results keyed by judge name."""
    if not trace.output.strip():
        # amp's _requires_response_output: scoring a blank response is misleading.
        return jc.skip_all(judges, "Trace has no response output")

    try:
        response = client.system_one(
            state=jc.build_state(trace),
            questions={j.name: build_question(j, task) for j in judges},
        )
    except Exception as exc:  # noqa: BLE001 - surface as a skip, like amp does
        return jc.skip_all(judges, f"Jev call failed: {exc}")

    results: Dict[str, EvalResult] = {}
    for judge in judges:
        answer = response.scores.get(judge.name)
        if answer is None:
            results[judge.name] = EvalResult.skip("Jev returned no answer for this question")
        else:
            results[judge.name] = to_result(judge, answer)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _client(api_key: Optional[str]) -> TypeSafeClient:
    key = api_key or os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
    if not key:
        sys.exit("Set JEV_API_KEY (or TYPESAFE_API_KEY) before running.")
    return TypeSafeClient(api_key=key)


def _print_comparison(rows: List[Dict[str, Any]], judges: Sequence[JudgeSpec]) -> None:
    width = 18
    header = f"{'trace':<14}" + "".join(f"{j.name:>{width}}" for j in judges)
    print(header)
    print(f"{'':<14}" + "".join(f"{'jev / llm':>{width}}" for _ in judges))
    print("-" * len(header))

    deltas: List[float] = []
    for row in rows:
        line = f"{row['trace_id'][:12]:<14}"
        for judge in judges:
            jev, llm = row["results"][judge.name], row["baseline"][judge.name]
            line += f"{jc.cell(jev) + ' / ' + jc.cell(llm):>{width}}"
            if not jev.is_skipped and not llm.is_skipped:
                deltas.append(abs(jev.score - llm.score))
        print(line)

    if deltas:
        within = sum(1 for d in deltas if d <= 0.25) / len(deltas)
        print()
        print(
            f"{len(deltas)} paired scores  "
            f"mean |jev - llm| = {sum(deltas) / len(deltas):.3f}  "
            f"within one rubric step = {within:.0%}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = jc.base_parser(__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run llm_judge.py's prompt and show both scores",
    )
    parser.add_argument(
        "--compare-model", default=None, help="Model for the --compare baseline"
    )
    parser.add_argument("--api-key", help="Jev API key (default: $JEV_API_KEY)")
    args = parser.parse_args(argv)

    judges = jc.select_judges(parser, args.judges)
    corpus = jc.load_corpus(args)

    if args.compare:
        import llm_judge

        baseline_model = args.compare_model or llm_judge.DEFAULT_MODEL

    rows: List[Dict[str, Any]] = []
    with _client(args.api_key) as client:
        for trace, task in corpus:
            row = {
                "trace_id": trace.trace_id,
                "trace_start_time": trace.start_time,
                "task_id": task.task_id if task else None,
                "results": score_trace(client, trace, judges, task),
            }
            if args.compare:
                row["baseline"] = llm_judge.score_trace(
                    trace, judges, task, baseline_model
                )
            rows.append(row)

    if args.json:
        jc.print_json(rows, judges)
    elif args.compare:
        _print_comparison(rows, judges)
    else:
        jc.print_table(rows, judges)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
