# Evaluation summary

48 seeded tasks (6 scenario families x 8 seeds), planted ground truth, 400 paired solver replays per context.

## Negative-value detection

Predicted negative = verdict ``liability`` (measured value <= -0.02). True negatives planted: 72.

| metric | value |
|---|---|
| precision | 1.000 |
| recall | 1.000 |
| F1 | 1.000 |
| true positives | 72 |
| false positives | 0 |
| false negatives | 0 |

Recall by planted negative class:

- contradictory: 1.000
- distractor: 1.000
- stale: 1.000

Detection vs replay budget (the same audits, priced with fewer paired replays):

| replays | precision | recall | F1 | false alarms | missed |
|---|---|---|---|---|---|
| 25 | 0.889 | 1.000 | 0.941 | 9 | 0 |
| 50 | 0.828 | 1.000 | 0.906 | 15 | 0 |
| 100 | 0.935 | 1.000 | 0.966 | 5 | 0 |
| 400 | 1.000 | 1.000 | 1.000 | 0 | 0 |

Coarse replays quantise the success rates, so inert segments near the liability threshold get false alarms; planted liabilities are never missed at any budget measured.

## Value accuracy

Mean absolute error of measured leave-one-out values vs the solver's analytic marginals: **0.0100** (success-probability units).

## Redundancy masking (redundant_key tasks)

Each copy of the duplicated key fact prices **0.070** on average alone; the pair prices **0.542** jointly. Leave-one-out pricing alone would call both copies nearly worthless.

## Budget packing: mean measured success

| strategy | 25% | 50% | 75% | 100% | retained @50% |
|---|---|---|---|---|---|
| value | 0.544 | 0.842 | 0.872 | 0.873 | 1.788 |
| relevance | 0.556 | 0.673 | 0.589 | 0.471 | 1.429 |
| recency | 0.239 | 0.162 | 0.248 | 0.471 | 0.343 |
| oracle | 0.563 | 0.842 | 0.873 | 0.874 | 1.789 |

Full context (everything kept, no budget): mean success **0.471**. Retained @50% is the ratio of aggregate means; above 1.0 means the budget cut helped by evicting liabilities.

## Per-scenario success at the 50% budget

| scenario | full | value | relevance | recency | oracle |
|---|---|---|---|---|---|
| bloated_history | 0.647 | 0.871 | 0.836 | 0.178 | 0.872 |
| contradictory_tool | 0.482 | 0.723 | 0.718 | 0.284 | 0.723 |
| distractor_heavy | 0.115 | 0.865 | 0.573 | 0.056 | 0.865 |
| redundant_key | 0.703 | 0.865 | 0.783 | 0.163 | 0.865 |
| stale_instruction | 0.240 | 0.864 | 0.314 | 0.212 | 0.865 |
| standard_mix | 0.638 | 0.864 | 0.813 | 0.078 | 0.864 |
