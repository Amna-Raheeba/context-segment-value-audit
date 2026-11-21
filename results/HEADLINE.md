# Headline results

Across 48 seeded tasks (6 scenario families x 8 seeds), planted ground truth:

- At a 50% token budget, value-ranked packing scores **0.842** mean success vs **0.471** for the full context -- a retained ratio of **1.788**. Keeping everything is worse than keeping the right half, because the full context keeps its liabilities. Relevance-ranked packing scores 0.673 (retained 1.429), recency-ranked 0.162 (retained 0.343), and the ground-truth oracle 0.842 -- the measured-value packer matches the oracle at this budget
- Value-ranked packing at a 100% budget scores 0.873 vs 0.471 for the full context -- dropping the negative-value segments beats keeping everything, and relevance-ranked packing gets *worse* as its budget grows (0.673 at 50%, 0.471 at 100%) because a bigger budget admits more distractors
- Negative-value detection at 400 paired replays: precision **1.000**, recall **1.000**, F1 **1.000** (72 planted negatives). At 50 replays precision falls to 0.828 -- the failure mode is false alarms on inert segments, never missed liabilities (recall 1.000)
- Measured leave-one-out values vs analytic marginals: MAE **0.0100**
- Redundancy masking: a duplicated key fact prices 0.070 per copy on average, but 0.542 jointly (8 redundant-key tasks)

All numbers are measured on synthetic seeded data; see README section 7 for limits.
