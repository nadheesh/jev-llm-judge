# jev-llm-judge

Can a decision model replace an LLM as an eval judge?

This runs the same evaluator rubrics two ways against the same agent traces:

- **`llm_judge.py`** sends a prompt to a chat model and parses `{"explanation", "score"}` back out. This is how [WSO2 Agent Manager's `amp-evaluation`](https://github.com/wso2/agent-manager/tree/main/libs/amp-evaluation) library does it, and the prompt is reassembled from that library's source.
- **`jev_judge.py`** sends the same rubrics to [Jev](https://www.jev-llm.com/), TypeSafe AI's System One model, as typed `Score` questions. No text is generated. Jev returns a probability distribution over the rubric levels.

Both emit the same `EvalResult` contract, so the scores are directly comparable.

**Short answer:** yes for scoring, no for explaining. Jev tracks the LLM judge closely on direction, is markedly more stable across repeat runs, is less generous, and tells you when it is unsure. It cannot tell you *why*, and on one of the three rubrics it fails to discriminate at all.

## Quick start

```bash
pip install -r requirements.txt

export JEV_API_KEY=...        # or TYPESAFE_API_KEY
export OPENAI_API_KEY=...     # only for llm_judge.py and --compare

python jev_judge.py           # Jev scores, five curated traces
python llm_judge.py           # the LLM baseline on the same five
python jev_judge.py --compare # both, side by side
```

Sample data is vendored under `data/`, so a clone runs offline and every run scores identical bytes.

## How a rubric becomes a typed question

`amp-evaluation`'s judges score 0.0 to 1.0 against a five-level rubric. The LLM sees the whole thing as prose and is asked to emit a float:

```
Scoring Rubric:
  0.0  = Not helpful at all; ignores the user's need...
  0.25 = Minimally helpful; touches on the topic but...
  0.5  = Somewhat helpful; provides some useful content but...
  0.75 = Helpful; addresses the user's need well with...
  1.0  = Highly helpful; directly and fully assists the user...
```

Jev takes the same five descriptions as the ordered `criteria` of a `Score` question and returns a probability-weighted level in `0..4`. Divide by 4 and you are back on amp's scale:

```python
Score(
    instructions="...your sole criterion is HELPFULNESS...",
    criteria=[  # the five rubric levels, verbatim
        "Not helpful at all; ignores the user's need...",
        ...
    ],
)
```

Two structural consequences:

- **One call per trace, not one per evaluator.** Jev takes a single `state` and a *map* of questions, so all three judges score a trace in one round trip. The LLM baseline needs three.
- **You get a confidence value.** Jev reports how peaked the distribution is. The LLM judge returns a confident-sounding paragraph whether or not it is confident.

## The traces

`data/sample_traces.json` is `amp-evaluation`'s own sample file: 14 traces, of which 10 carry a real exchange. The other 4 are bare embedding spans and one CrewAI trace whose "input" is a 148k-character Python `repr` of framework objects rather than anything a person said.

A note for anyone building on this file: only 3 of the 14 traces populate `input`/`output` at the trace root. The rest are rooted at `POST /chat` and keep the conversation on a nested `*.workflow` span, which is where `amp-evaluation`'s real trace parser looks. Reading the root alone gets you 3 traces; reading the spans gets you 10.

Five are curated as the default set, chosen to spread the rubric rather than to flatter either judge. Two pairs are deliberately close:

| trace | what happened |
|---|---|
| `fc5513186f8d` | substantive travel recommendations, but silently ignores "I'm going next week" |
| `789a4cc3a165` | hotel tool failed; agent gives generic neighborhood advice dressed up as help |
| `7ec82d8703e4` | hotel tool failed; agent gives the user nothing, "try again later" |
| `3c6c9d5f6890` | vague request ("I wanna go to USA"); agent correctly asks for details |
| `59ab6f6cea2c` | agent starts a plan, then stops to ask for the departure city |

`789a4cc3` and `7ec82d87` both follow a tool failure, one salvaging it and one not. `7ec82d87` and `3c6c9d5f` both leave the user without a booking, one by failing and one by correctly asking a question. A judge worth trusting should separate each pair.

Run `--all` to score all ten.

## Results

Median of three runs. Raw per-run scores are in `results/three-run-scores.json`.

```
trace                helpfulness           clarity      completeness
                       jev / llm         jev / llm         jev / llm
fc5513186f8d         0.70 / 1.00       0.88 / 1.00       0.91 / 0.75
789a4cc3a165         0.50 / 0.75       0.83 / 0.75       0.77 / 1.00
7ec82d8703e4         0.18 / 0.00       0.83 / 0.50       0.04 / 0.00
3c6c9d5f6890         0.65 / 1.00       0.98 / 1.00       0.83 / 1.00
59ab6f6cea2c         0.45 / 0.50       0.83 / 0.75       0.34 / 0.50
```

Baseline is `gpt-4o-mini` at `temperature=0.0` with `response_format={"type": "json_object"}`.

### 1. Stability

Three runs, identical inputs, temperature 0 on both sides:

| | mean spread over 3 runs | worst cell | cells identical across all 3 |
|---|---|---|---|
| Jev | **0.009** | 0.023 | 1 / 15 |
| LLM judge | **0.117** | 0.750 | 10 / 15 |

The LLM judge reproduces itself exactly in two thirds of cells and then swings wildly in the rest. Completeness on `3c6c9d5f` came back `[1.00, 1.00, 0.25]` on identical input. Jev never moved more than 0.023 anywhere, and its small jitter is continuous rather than a jump between rubric levels.

This is the clearest result here. If you are gating a release on an eval score, a judge that sometimes returns 0.25 where it returned 1.00 twice is not a gate.

### 2. Jev is stricter

Jev scores lower on 9 of 15 cells, and the effect is concentrated where it matters: **4 of 5 on helpfulness**. On clarity it is actually higher 3 times out of 5, for reasons covered below. The pattern on helpfulness is that the LLM judge rewards tone and formatting:

- On `3c6c9d5f`, the agent answers "I wanna go to USA" with six clarifying questions. `gpt-4o-mini` scores helpfulness **1.00**. amp's 1.0 level reads "directly and fully assists the user". It assisted nobody yet. Asking was the right move, but the rubric does not say 1.0. Jev's 0.65 is what the rubric text supports.
- On `789a4cc3`, the tool failed and the user got no accommodations, just a polished list of neighborhoods and a pointer to Booking.com. The LLM gives 0.75. Jev gives 0.50, which is amp's "provides some useful content but the user would still need significant additional help". That is exactly what happened.

Both judges correctly rank `7ec82d87` worst, so the disagreement is about calibration, not direction.

### 3. Clarity is broken on the Jev side

Jev's clarity column barely moves: 0.88, 0.83, 0.83, 0.98, 0.83. It gives the same 0.83 to a 1,200-word structured itinerary and to a 276-character "try again later" brush-off. The LLM judge drops that brush-off to 0.50 and is the better judge here.

The likely cause is that clarity is close to unconditionally satisfied for any fluent model output, so the rubric has almost no signal to separate on. It is a reminder that a rubric which works as an LLM prompt does not automatically work as a typed question, and each one needs checking.

### 4. Confidence points at the weak scores

Jev reports confidence per answer. In practice it flags the scores that are hardest to defend:

```
789a4cc3a165  completeness  score 0.77  confidence 0.20
3c6c9d5f6890  clarity       score 0.98  confidence 0.95
```

The 0.20 is on a score I would argue is wrong. The 0.95 is on one that is obviously right. That signal has no equivalent on the LLM side, which returns a fluent paragraph of justification at a uniform pitch of certainty.

A practical use: route low-confidence traces to human review instead of trusting the number.

## What you lose

The explanation. Jev does not generate text, so there is no argument to read, no audit trail, and nothing to paste into a bug report. The `explanation` field carries the rubric level and the confidence instead:

```json
{
  "evaluator": "helpfulness",
  "score": 0.5,
  "passed": true,
  "explanation": "jev rubric level 1.98/4 (confidence 0.83) [model=jev]"
}
```

If the eval's job is to gate CI or track a metric over time, that is enough. If a human reads every result to understand a regression, it is not.

## Caveats

Read the numbers as a demonstration, not a benchmark.

- Five traces, three runs, one baseline model. The agreement percentages are illustrative. Nothing here is significant.
- All five traces come from one LangGraph travel agent. Other domains may behave differently.
- The baseline is `gpt-4o-mini`. A larger judge model would likely be both better calibrated and more stable. `--compare-model` takes anything the OpenAI SDK accepts.
- Jev's scores are continuous while the LLM lands on the coarse 0.25 steps, so exact equality was never achievable. "Within one rubric step" is the meaningful comparison.

## Files

| file | what it is |
|---|---|
| `judge_common.py` | the shared contract: `Trace`, `Task`, `EvalResult`, the three rubrics, trace loading, output formatting |
| `jev_judge.py` | rubrics as Jev `Score` questions, plus `--compare` |
| `llm_judge.py` | the same rubrics as an LLM prompt, amp's own shape |
| `data/` | unmodified sample traces and dataset from `amp-evaluation` |
| `results/` | raw scores from the three-run stability measurement |

The rubric text lives in exactly one place, `judge_common.py`, so a score difference is a difference between the two judges and not between two drifting copies of a prompt.

## License

Apache 2.0. See `LICENSE`.

The evaluator rubrics and the sample data are derived from [wso2/agent-manager](https://github.com/wso2/agent-manager), also Apache 2.0. See `NOTICE` for the specific files and what was taken from each.
