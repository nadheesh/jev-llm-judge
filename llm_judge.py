#!/usr/bin/env python3
"""
llm_judge, amp-evaluation's LLM-as-judge evaluators, run the way amp runs them.

This is the baseline. It rebuilds the prompt that
`LLMAsJudgeEvaluator.build_prompt()` produces (instructions, the trace text
inlined, the numbered rubric, then amp's own output-format block), sends it to
a chat model, and parses the {"explanation", "score"} JSON back out.

One prompt per evaluator per trace, which is amp's own shape: five evaluators
over the five curated traces is twenty-five round trips.

Run `jev_judge.py` for the same rubrics on Jev, or `jev_judge.py --compare`
to see both side by side.

Usage:
    export OPENAI_API_KEY=...
    python llm_judge.py                        # score the five curated traces
    python llm_judge.py --all                  # every trace with a real exchange
    python llm_judge.py --json                 # EvaluatorScore-shaped JSON
    python llm_judge.py --model gpt-4o --limit 3

Requires: pip install openai
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Sequence

import judge_common as jc
from judge_common import EvalResult, JudgeSpec, Task, Trace

DEFAULT_MODEL = "gpt-4o-mini"

# Copied from LLMAsJudgeEvaluator._OUTPUT_INSTRUCTIONS. The Markdown clause is
# kept out: it shapes the explanation, which these scripts compare on score.
OUTPUT_INSTRUCTIONS = """

First provide your reasoning, then your score. Respond with a JSON object:
{
  "explanation": "<your step-by-step analysis>",
  "score": <float between 0.0 and 1.0, where 0.0 is the worst possible and 1.0 is the best possible>
}"""


def build_prompt(judge: JudgeSpec, trace: Trace, task: Optional[Task] = None) -> str:
    """Reassemble amp's build_prompt(): instructions, trace text, rubric, format."""
    rubric = "\n".join(
        f"  {label} = {text}" for label, text in zip(jc.RUBRIC_LABELS, judge.rubric)
    )
    context = ""
    if judge.needs_evidence:
        context += f"\nEvidence Available to the Agent:\n{trace.format_evidence()}\n"
    if judge.needs_trajectory:
        context += (
            f"\nTotal Steps: {len(trace.tool_calls)}\n"
            f"Execution Steps:\n{trace.format_steps()}\n"
        )
    return (
        f"{judge.instructions_for(task)}\n\n"
        f"User Query: {trace.input}\n"
        f"Agent Response: {trace.output}\n"
        f"{context}\n"
        f"Scoring Rubric:\n{rubric}"
        f"{OUTPUT_INSTRUCTIONS}"
    )


def score_trace(
    trace: Trace,
    judges: Sequence[JudgeSpec],
    task: Optional[Task] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, EvalResult]:
    """One chat completion per judge. Returns results keyed by judge name."""
    try:
        from openai import OpenAI
    except ImportError:
        return jc.skip_all(judges, "openai is required: pip install openai")

    client = OpenAI()
    results: Dict[str, EvalResult] = {}
    for judge in judges:
        skipped = jc.preflight(trace, judge)
        if skipped is not None:
            results[judge.name] = skipped
            continue
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": build_prompt(judge, trace, task)}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            payload = json.loads(completion.choices[0].message.content)
            results[judge.name] = EvalResult(
                score=float(payload["score"]),
                explanation=f"{payload.get('explanation', '')} [model={model}]",
            )
        except Exception as exc:  # noqa: BLE001 - amp skips on judge failure too
            results[judge.name] = EvalResult.skip(f"LLM judge failed: {exc}")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = jc.base_parser(__doc__.split("\n")[1].strip())
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chat model to judge with")
    args = parser.parse_args(argv)

    judges = jc.select_judges(parser, args.judges)
    rows = [
        {
            "trace_id": trace.trace_id,
            "trace_start_time": trace.start_time,
            "task_id": task.task_id if task else None,
            "results": score_trace(trace, judges, task, args.model),
        }
        for trace, task in jc.load_corpus(args)
    ]

    (jc.print_json if args.json else jc.print_table)(rows, judges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
