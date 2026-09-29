# jev-llm-judge

Can a decision model replace an LLM as an eval judge?

This runs the same evaluator rubrics two ways against the same agent traces:

- **`llm_judge.py`** sends a prompt to a chat model and parses `{"explanation", "score"}` back out. This is how [WSO2 Agent Manager's `amp-evaluation`](https://github.com/wso2/agent-manager/tree/main/libs/amp-evaluation) library does it, and the prompt is reassembled from that library's source.
- **`jev_judge.py`** sends the same rubrics to [Jev](https://www.jev-llm.com/), TypeSafe AI's System One model, as typed `Score` questions. No text is generated. Jev returns a probability distribution over the rubric levels.

Both emit the same `EvalResult` contract, so the scores are directly comparable.

Five of amp's built-in evaluators are ported: **helpfulness**, **clarity**, **completeness**, **groundedness** and **path_efficiency**. The last two read the tool trajectory, not just the reply.

**Short answer:** yes for scoring, no for explaining. Jev never moves a score by more than 0.04 across repeat runs, is less generous than the LLM judge on the response rubrics, and is sharply better on path efficiency, where the LLM grades the outcome instead of the path. It cannot tell you *why*, and on one of the five rubrics it fails to discriminate at all.

## Quick start

```bash
pip install -r requirements.txt

cp .env.example .env          # then paste your keys in; both scripts read it
# or export JEV_API_KEY / OPENAI_API_KEY directly, which takes precedence

python jev_judge.py           # Jev scores, five evaluators, five curated traces
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

Median of three runs. Every per-run score is in `results/three-run-scores.json`, and the Jev runs are reproduced in full below.

```
trace                helpfulness           clarity      completeness      groundedness   path_efficiency
                       jev / llm         jev / llm         jev / llm         jev / llm         jev / llm
fc5513186f8d         0.69 / 1.00       0.89 / 1.00       0.92 / 0.75       0.21 / 0.50       0.74 / 0.50
789a4cc3a165         0.50 / 0.75       0.83 / 1.00       0.76 / 1.00       0.37 / 0.25       0.34 / 0.00
7ec82d8703e4         0.18 / 0.25       0.83 / 0.50       0.04 / 0.00       0.85 / 1.00       0.81 / 0.00
3c6c9d5f6890         0.65 / 1.00       0.98 / 1.00       0.83 / 1.00       skip / skip       1.00 / 1.00
59ab6f6cea2c         0.46 / 0.50       0.83 / 0.75       0.33 / 0.50       skip / skip       0.94 / 0.50
```

Baseline is `gpt-4o-mini` at `temperature=0.0` with `response_format={"type": "json_object"}`.

Groundedness shows `skip` on the two traces with no tool calls. That is amp's own rule: `GroundednessEvaluator.evaluate()` skips rather than scoring zero when there is nothing to check claims against.

### 1. Stability

Three runs, identical inputs, temperature 0 on both sides, 23 scored cells per side:

| | mean spread over 3 runs | worst cell | cells identical across all 3 |
|---|---|---|---|
| Jev | **0.013** | 0.038 | 0 / 23 |
| LLM judge | **0.054** | 0.250 | 18 / 23 |

The two judges are unstable in different ways. Jev jitters continuously and never by more than 0.04, which never changes a verdict. The LLM judge reproduces itself exactly in 18 of 23 cells and then jumps a whole rubric level in the rest.

Worth being honest about: an earlier three-run measurement over the first three evaluators put the LLM's mean spread at 0.117 with one cell swinging 0.750. That did not recur here. The LLM's instability is itself unstable, which is the practical problem with it, but it also means neither number should be quoted as *the* figure.

#### Every Jev run

All three runs, per trace and evaluator, so you can see the jitter rather than take the summary on trust:

| trace | evaluator | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|---|
| `fc5513186f8d` | helpfulness | 0.695 | 0.690 | 0.693 | 0.005 |
| `fc5513186f8d` | clarity | 0.885 | 0.890 | 0.885 | 0.005 |
| `fc5513186f8d` | completeness | 0.915 | 0.922 | 0.910 | 0.012 |
| `fc5513186f8d` | groundedness | 0.210 | 0.210 | 0.215 | 0.005 |
| `fc5513186f8d` | path_efficiency | 0.730 | 0.740 | 0.738 | 0.010 |
| `789a4cc3a165` | helpfulness | 0.492 | 0.502 | 0.502 | 0.010 |
| `789a4cc3a165` | clarity | 0.828 | 0.828 | 0.830 | 0.002 |
| `789a4cc3a165` | completeness | 0.745 | 0.757 | 0.775 | 0.030 |
| `789a4cc3a165` | groundedness | 0.367 | 0.350 | 0.388 | 0.038 |
| `789a4cc3a165` | path_efficiency | 0.350 | 0.323 | 0.340 | 0.027 |
| `7ec82d8703e4` | helpfulness | 0.177 | 0.182 | 0.180 | 0.005 |
| `7ec82d8703e4` | clarity | 0.830 | 0.838 | 0.830 | 0.008 |
| `7ec82d8703e4` | completeness | 0.035 | 0.043 | 0.035 | 0.007 |
| `7ec82d8703e4` | groundedness | 0.850 | 0.843 | 0.858 | 0.015 |
| `7ec82d8703e4` | path_efficiency | 0.780 | 0.818 | 0.805 | 0.037 |
| `3c6c9d5f6890` | helpfulness | 0.652 | 0.660 | 0.647 | 0.013 |
| `3c6c9d5f6890` | clarity | 0.988 | 0.983 | 0.983 | 0.005 |
| `3c6c9d5f6890` | completeness | 0.850 | 0.835 | 0.823 | 0.027 |
| `3c6c9d5f6890` | groundedness | skip | skip | skip | - |
| `3c6c9d5f6890` | path_efficiency | 0.998 | 0.995 | 0.998 | 0.003 |
| `59ab6f6cea2c` | helpfulness | 0.455 | 0.445 | 0.455 | 0.010 |
| `59ab6f6cea2c` | clarity | 0.828 | 0.835 | 0.838 | 0.010 |
| `59ab6f6cea2c` | completeness | 0.333 | 0.338 | 0.333 | 0.005 |
| `59ab6f6cea2c` | groundedness | skip | skip | skip | - |
| `59ab6f6cea2c` | path_efficiency | 0.925 | 0.940 | 0.935 | 0.015 |

The largest single movement anywhere is 0.038, on groundedness for `789a4cc3`. Every score stays inside its rubric level across all three runs.

### 2. Path efficiency is where the gap is widest

This evaluator scores the trajectory, not the answer: redundant calls, loops, wasted work. The traces give it something real to find. `789a4cc3` called `search_hotels` five times, once per city, and got `ValueError` every time. `7ec82d87` called two different tools once each and both failed.

| trace | trajectory | jev | llm |
|---|---|---|---|
| `789a4cc3a165` | same tool 5x, all failed | 0.34 | 0.00 |
| `7ec82d8703e4` | 2 distinct tools, both failed | 0.81 | 0.00 |
| `3c6c9d5f6890` | no tool calls, asked a question instead | 1.00 | 1.00 |

The LLM judge gives **0.00 to both failure traces**, collapsing a genuinely redundant path and a minimal one into the same score. It is grading the outcome. But the rubric asks about redundancy, loops and irrelevant steps, and a path of two distinct calls is not inefficient just because the tools were broken. Jev separates them 0.34 against 0.81, which is the distinction the rubric was written to make.

If you are using this evaluator to find agents that waste tokens looping, the LLM judge will bury them among every agent that merely hit a failing dependency.

### 3. Groundedness inverts the ranking, correctly

Both judges agree on the direction here, and the result is the most counterintuitive in the set:

- `7ec82d87`, the **least** helpful response (0.18), is the **most** grounded (0.85). It says "there is a technical issue, try again later" and asserts no facts at all, so it has nothing unsupported to be wrong about.
- `fc551318`, the most helpful (0.69), is the **least** grounded (0.21). Its confident specifics about the Alhambra and Sierra Nevada trace back to one failed `search_trip_recommendations` call and a single web search, so most of the detail is the model's own knowledge presented as retrieved fact.

That is the evaluator working. It is also a caution about reading any single score as quality: an agent can raise its groundedness by saying less. Groundedness and helpfulness are only meaningful together.

### 4. Jev is stricter on the response rubrics

Jev scores lower on **all 5** helpfulness cells, and on 3 of 5 for both clarity and completeness. The direction reverses only on path_efficiency, where Jev is higher on 4 of 5 for the reason above. The pattern on the response rubrics is that the LLM judge rewards tone and formatting:

- On `3c6c9d5f`, the agent answers "I wanna go to USA" with six clarifying questions. `gpt-4o-mini` scores helpfulness **1.00**. amp's 1.0 level reads "directly and fully assists the user". It assisted nobody yet. Asking was the right move, but the rubric does not say 1.0. Jev's 0.65 is what the rubric text supports.
- On `789a4cc3`, the tool failed and the user got no accommodations, just a polished list of neighborhoods and a pointer to Booking.com. The LLM gives 0.75. Jev gives 0.50, which is amp's "provides some useful content but the user would still need significant additional help". That is exactly what happened.

Both judges rank `7ec82d87` worst on helpfulness, so the disagreement is about calibration, not direction.

### 5. Clarity is flat on the Jev side

Jev's clarity column barely moves: 0.89, 0.83, 0.83, 0.98, 0.83. It gives the same 0.83 to a 1,200-word structured itinerary and to a 276-character "try again later" brush-off. The LLM judge drops that brush-off to 0.50 and is the better judge here.

The likely cause is that clarity is close to unconditionally satisfied for any fluent model output, so the rubric has almost nothing to separate on. A rubric that works as an LLM prompt does not automatically work as a typed question, and each one needs checking. Of the five here, clarity is the one that did not survive the translation.

### 6. Confidence points at the weak scores

Jev reports confidence per answer, and in practice it flags the scores that are hardest to defend. On an earlier run, completeness for `789a4cc3` came back at score 0.77 with confidence **0.20**, which was the one score in that batch I would argue is wrong. Clarity for `3c6c9d5f` came back 0.98 at confidence **0.95**, which is obviously right.

There is no equivalent on the LLM side, which returns a fluent paragraph of justification at a uniform pitch of certainty whether it is sure or not.

A practical use: route low-confidence traces to human review instead of trusting the number.

### Cost shape

Jev batches judges that read the same state into one call. All five evaluators over one trace cost **three** Jev calls (the three response-only judges together, then groundedness and path_efficiency, which each need a field of their own) against **five** LLM calls. Over the five-trace set that is 15 calls versus 25.

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

- Five traces, three runs, one baseline model. Nothing here is statistically significant, and the stability figures moved noticeably between measurements.
- `path_efficiency` scores a trace with **zero** tool calls as near-perfect (1.00 and 0.94). Defensible, since asking a question beats guessing, but it means the score rewards inaction and should be read next to helpfulness.
- All five traces come from one LangGraph travel agent. Other domains may behave differently.
- The baseline is `gpt-4o-mini`. A larger judge model would likely be both better calibrated and more stable. `--compare-model` takes anything the OpenAI SDK accepts.
- Jev's scores are continuous while the LLM lands on the coarse 0.25 steps, so exact equality was never achievable. "Within one rubric step" is the meaningful comparison.

## Files

| file | what it is |
|---|---|
| `judge_common.py` | the shared contract: `Trace`, `ToolCall`, `EvalResult`, the five rubrics, span parsing, output formatting |
| `jev_judge.py` | rubrics as Jev `Score` questions, plus `--compare` |
| `llm_judge.py` | the same rubrics as an LLM prompt, amp's own shape |
| `data/` | unmodified sample traces and dataset from `amp-evaluation` |
| `results/` | raw scores from the three-run stability measurement |

The rubric text lives in exactly one place, `judge_common.py`, so a score difference is a difference between the two judges and not between two drifting copies of a prompt.

## License

Apache 2.0. See `LICENSE`.

The evaluator rubrics and the sample data are derived from [wso2/agent-manager](https://github.com/wso2/agent-manager), also Apache 2.0. See `NOTICE` for the specific files and what was taken from each.
