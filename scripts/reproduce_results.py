"""Headline-drift guard.

Re-runs the full seeded evaluation and compares the aggregate
headline numbers against the shipped results/evaluation.json.
Exits non-zero if any headline value moved by more than the
tolerance -- i.e. if the code no longer reproduces the published
results.

Run from the repository root:  python scripts/reproduce_results.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_segment_value.evaluate import run_evaluation
from context_segment_value.packing import BUDGETS, STRATEGIES

TOLERANCE = 0.02


def headline_values(results):
    det = results["detection"]
    values = {
        "precision": det["precision"],
        "recall": det["recall"],
        "f1": det["f1"],
        "value_mae": results["value_mae"],
        "mean_full_success": results["mean_full_success"],
        "masking_individual": results["masking"]["mean_individual"],
        "masking_joint": results["masking"]["mean_joint"],
    }
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            key = "%.2f" % budget
            values["packing_%s_%s" % (strategy, key)] = \
                results["packing"][strategy][key]
        values["retained_%s" % strategy] = \
            results["packing"][strategy]["retained_at_50"]
    for n_draws, entry in results["detection_by_draws"].items():
        values["draws_%s_precision" % n_draws] = entry["precision"]
        values["draws_%s_recall" % n_draws] = entry["recall"]
    return values


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shipped_path = os.path.join(root, "results", "evaluation.json")
    with io.open(shipped_path, encoding="utf-8") as fh:
        shipped = json.load(fh)
    fresh = run_evaluation()
    expected = headline_values(shipped)
    actual = headline_values(fresh)
    failures = []
    for key in sorted(expected):
        if key not in actual:
            failures.append("%s missing from fresh run" % key)
        elif abs(expected[key] - actual[key]) > TOLERANCE:
            failures.append(
                "%s: shipped %.4f vs fresh %.4f"
                % (key, expected[key], actual[key])
            )
    if failures:
        print("HEADLINE DRIFT DETECTED:")
        for failure in failures:
            print("  " + failure)
        return 1
    print("All %d headline values reproduce within %.2f."
          % (len(expected), TOLERANCE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
