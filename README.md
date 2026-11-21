# Context Segment Value Audit

## 1. What it is

An agent's context is usually treated as an asset: more retrieved
chunks, more history, more tool output -- more help. Some of it is
not help. A distractor chunk that shares the query's vocabulary but
states the wrong value, a stale instruction left next to its
replacement, a tool output that contradicts the key fact in a
confident machine format -- these segments measurably *reduce* the
chance the task succeeds, and they ride along in production prompts
because nothing prices them.

This project prices every segment of an agent's context by its
**causal leave-one-out effect on task success**: replay the task
with the full context, replay it with one segment removed, and the
difference is the segment's value. Segments are reported with a
value, a value density (value per token), and a verdict -- asset,
inert, or liability -- and the liabilities are collected into a
**negative-value register**: the list of context that is worse than
no context. The prices then drive a budget packer that is compared
against the two heuristics real systems actually use (relevance
ranking and recency ranking) and against a ground-truth oracle.

Everything is offline, deterministic and seeded. The "agent" is a
synthetic solver whose per-segment contributions are planted ground
truth, so the audit instrument itself is scored -- detection
precision/recall, value error, packing success -- instead of merely
demonstrated.

## 2. The novel mechanism

Four pieces, none of them a trained model:

1. **Paired leave-one-out pricing.** Each task is replayed 400
   times under *common random numbers*: the full context and every
   one-segment-poorer context face the identical draws, so a
   segment's measured value is a paired causal contrast, not two
   noisy samples. Measured values track the solver's analytic
   marginals to an MAE of 0.0100 success-probability units.
2. **The negative-value register.** Pricing is signed. A segment
   whose removal *raises* success is a liability, whatever its role
   or apparent authority -- in the demo task the worst segment in
   the context is an 8-token tool output (value -0.422, -0.0528 per
   token) that contradicts the retrieved key fact.
3. **A redundancy probe that prices pairs jointly.** Leave-one-out
   pricing has a known blind spot: two copies of a key fact each
   price near zero (remove one, the other carries the task), so a
   naive audit would call both worthless. The audit detects
   near-duplicate pairs by text similarity and prices them jointly,
   exposing the masking instead of hiding it (section 5).
4. **Value-ranked packing, honestly compared.** Under a token
   budget the packer fills by measured value density and refuses
   liabilities at any budget. It is scored against relevance-ranked
   (TF-IDF) and recency-ranked packing and against an oracle that
   ranks by ground-truth value -- the ceiling that shows whether
   *measured* prices are good enough to act on.

## 3. Prior art and the differentiator

- **LLMLingua** and the prompt-compression family prune tokens by
  perplexity: they score how *predictable* text is, with a model,
  and never measure a segment's causal effect on task success.
  Predictability is not value -- a short contradictory tool output
  is highly predictable and strongly negative.
- **Prompt Compression Benchmarker** compares existing compressors
  on live APIs. It prices *methods*, not segments, and computes no
  per-segment causal value or negative-value register.
- **Citation-ablation faithfulness verification** (a sibling
  project) ablates citations inside a RAG *answer* to judge whether
  the answer is faithful to its sources. This tool ablates segments
  inside an agent's *input* to price context economics -- a
  different object, a different question, a different output.

The differentiator: causal per-segment pricing of an agent's input
context, with a signed register of negative-value segments and a
joint-pricing probe for the redundancy case where marginal pricing
lies -- all validated offline against planted ground truth.

## 4. How the audit works

1. **Generate** a task from one of six scenario families
   (`standard_mix`, `stale_instruction`, `contradictory_tool`,
   `redundant_key`, `bloated_history`, `distractor_heavy`), each
   assembling a context the way real prompts go wrong, with planted
   per-segment contributions.
2. **Solve** with the seeded solver: success probability is a
   logistic function of the present segments' contributions, with a
   duplicate discount (a fact retrieved twice does not help twice).
3. **Price** every segment by paired leave-one-out replay, compute
   value density, assign verdicts (asset >= +0.02, liability
   <= -0.02, else inert), and run the duplicate probe.
4. **Pack** the context under budgets of 25-100% of its tokens by
   value, relevance, recency and oracle rankings, and replay each
   packed context for its measured success.

## 5. Measured results

48 seeded tasks (6 scenario families x 8 seeds), 400 paired replays
per context, planted ground truth throughout. Full tables in
`results/evaluation_summary.md`; regeneration guarded by
`scripts/reproduce_results.py`.

**Budget packing -- mean measured success:**

| strategy | 25% | 50% | 75% | 100% | retained @50% |
|---|---|---|---|---|---|
| value | 0.544 | 0.842 | 0.872 | 0.873 | 1.788 |
| relevance | 0.556 | 0.673 | 0.589 | 0.471 | 1.429 |
| recency | 0.239 | 0.162 | 0.248 | 0.471 | 0.343 |
| oracle | 0.563 | 0.842 | 0.873 | 0.874 | 1.789 |

Full context (everything kept): **0.471**.

- **Keeping the right half beats keeping everything.** At a 50%
  token budget, value-ranked packing scores 0.842 mean success
  against 0.471 for the full context -- a retained ratio of 1.788.
  Retention above 1.0 is not a typo: the budget cut *helps* because
  the packer evicts liabilities the full context keeps. At a 100%
  budget the value packer still scores 0.873 vs 0.471, purely by
  refusing negative-value segments.
- **Measured prices match ground-truth prices.** The value packer
  tracks the oracle at every budget (0.842 vs 0.842 at 50%; 0.873
  vs 0.874 at 100%) -- the leave-one-out estimates are accurate
  enough that acting on them loses nothing against acting on the
  truth.
- **Relevance ranking degrades as its budget grows** (0.673 at 50%,
  0.471 at 100%): a bigger budget admits more distractors, which
  are built from the query's own vocabulary and outrank the neutral
  history on relevance in every task. In `distractor_heavy` tasks
  the full context succeeds 0.115 of the time; value packing at
  half the tokens scores 0.865.
- **Recency ranking is the worst strategy measured** (0.162 at
  50%): the most recent segments are history chatter, so
  "keep the latest turns" keeps the ballast and drops the facts.

**Negative-value detection** (predicted negative = verdict
liability; 72 planted negatives): precision **1.000**, recall
**1.000**, F1 **1.000** at 400 paired replays, with per-class
recall 1.000 for distractors, stale instructions and contradictory
tool outputs. The replay-budget sweep shows what a cheaper audit
costs:

| replays | precision | recall | false alarms | missed |
|---|---|---|---|---|
| 25 | 0.889 | 1.000 | 9 | 0 |
| 50 | 0.828 | 1.000 | 15 | 0 |
| 100 | 0.935 | 1.000 | 5 | 0 |
| 400 | 1.000 | 1.000 | 0 | 0 |

Coarse replays quantise the success rates, so inert segments near
the threshold collect false alarms -- but no planted liability is
missed at any budget measured. The failure mode of a cheap audit is
crying wolf on ballast, not sleeping through poison.

**Redundancy masking:** in the `redundant_key` tasks each copy of
the duplicated key fact prices **0.070** on average alone, while
the pair prices **0.542** jointly. Leave-one-out pricing alone
would report both copies as nearly worthless; the joint probe is
what keeps the audit honest here.

## 6. The negative-value register

The register is the audit's actionable output: segment id, role,
planted-class-agnostic measured harm, and token cost. From the
shipped demo task (`examples/sample_report.md`, real captured
output):

```
Negative-value register:
  s03 [tool_output / contradictory] value -0.422 over 8 tokens --
  removing it raises measured success
```

Three planted families land on the register, for three different
reasons a real prompt audit would care about: distractors
(lexically attractive, factually wrong), stale instructions
(superseded but never deleted), and contradictory tool outputs
(compact, authoritative, wrong). The register reports the measured
harm; it does not guess which family a segment belongs to.

## 7. Honest limits

- **The solver is synthetic.** Contributions are planted and
  additive on a logit scale, with one interaction (the duplicate
  discount). A real LLM's response to a segment is not additive,
  not stationary, and not known -- this tool validates the
  *instrument* (paired LOO pricing, the register, the packer)
  against ground truth, and its numbers are properties of this
  testbed, not of any production model.
- **Detection is exact partly because the testbed is separable.**
  Planted liabilities move the logit by 1.3-1.9 units, far above
  the measurement noise at 400 replays; the replay-budget sweep
  (section 5) is the honest part of the claim -- it shows exactly
  how detection degrades as measurement gets cheaper, and real
  segments with small true effects would sit in that degraded
  regime.
- **Leave-one-out value is marginal, not standalone.** A segment's
  price is its effect *in this context*: sigmoid saturation
  compresses prices when the context is already strong, and
  redundant segments price low individually. The joint probe
  covers planted duplicate pairs; arbitrary higher-order
  interactions (three-way redundancy, complementary pairs that
  only help together) are not probed.
- **Packing is greedy by density.** Greedy fill is not the optimal
  knapsack; the oracle comparison bounds the pricing error, not
  the packing error.
- **Token counts are whitespace counts** of template-generated
  text, and relevance uses a small TF-IDF implementation -- both
  are deliberate simplifications of production tokenisers and
  retrievers.

## 8. Usage

```bash
pip install pytest streamlit   # tests + dashboard; runtime is stdlib only

python -m context_segment_value demo                 # audit one task
python -m context_segment_value demo --scenario redundant_key --seed 2
python -m context_segment_value evaluate --out results
python examples/run_audit.py                          # audit the shipped demo task
python scripts/reproduce_results.py                   # headline-drift guard
streamlit run app.py                                  # dashboard
pytest -q                                             # 78 tests
```

## 9. Repository layout

```
context_segment_value/
  models.py      segments, tasks, planted ground truth, paired draws
  generate.py    six seeded scenario families + serialisation
  solver.py      seeded solver: contributions, duplicate discount
  value.py       leave-one-out pricing, duplicate probe, joint value
  relevance.py   TF-IDF baseline scorer (stdlib)
  packing.py     value / relevance / recency / oracle budget packing
  audit.py       verdicts, negative-value register, audit result
  evaluate.py    seeded evaluation against planted ground truth
  report.py      audit report + evaluation documents
  cli.py         demo / evaluate commands
data/            scenario descriptions + the shipped demo task
examples/        run_audit.py + real captured sample reports
notebooks/       01 demo run, 02 evaluation experiment (runnable .py)
scripts/         build_cases.py, reproduce_results.py (drift guard)
results/         evaluation.json/.csv, summary, headline
tests/           78 tests
app.py           Streamlit dashboard (Audit / Packing / Evaluation)
```

## 10. Reproducing the results

`python -m context_segment_value evaluate --out results`
regenerates every number in section 5 from the seeds alone.
`python scripts/reproduce_results.py` re-runs the evaluation and
fails if any of the 35 headline values drifts by more than 0.02
from the shipped `results/evaluation.json` -- and it runs as part
of the test suite, so the published numbers cannot silently rot.

## 11. License and author

MIT (see `LICENSE`). Author: Amna Raheeba.
