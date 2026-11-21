# Changelog

## 0.1.0 — 2026-11-22

- Initial release.
- Seeded task generator with six scenario families (standard mix,
  stale instruction, contradictory tool output, redundant key,
  bloated history, distractor heavy), each with planted per-segment
  ground-truth contributions.
- Seeded solver with known segment contributions, a duplicate
  discount, and common-random-number success replays.
- Leave-one-out causal segment pricing: per-segment value, value
  density per token, and a negative-value register (distractors,
  stale instructions, contradictory tool output).
- Near-duplicate joint-pricing probe that exposes redundancy
  masking (two copies of a key fact each price near zero alone).
- Budget packing comparison: value-ranked vs relevance-ranked
  (TF-IDF) vs recency-ranked vs oracle packing at 25–100% budgets.
- Seeded evaluation over 48 tasks with planted ground truth:
  negative-value detection precision/recall, value-estimation error
  vs analytic marginals, and success retained per packing strategy.
- CLI (`demo`, `evaluate`), Streamlit dashboard, reproduction
  drift guard.
