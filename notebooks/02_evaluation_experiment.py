"""Notebook 02 -- the evaluation experiment, reduced.

Runs the seeded evaluation on 3 seeds per scenario family (18
tasks) and prints the headline table. The shipped results in
results/ use 8 seeds (48 tasks); the reduced run shows the same
pattern at a third of the size.

Runnable as a plain script:  python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_segment_value.evaluate import run_evaluation
from context_segment_value.packing import BUDGETS, STRATEGIES


def main():
    results = run_evaluation(seeds=(0, 1, 2))
    det = results["detection"]
    print("Tasks evaluated: %d" % results["n_tasks"])
    print("Negative-value detection: precision %.3f, recall %.3f, "
          "F1 %.3f" % (det["precision"], det["recall"], det["f1"]))
    print("Value MAE vs analytic marginals: %.4f" % results["value_mae"])
    print("Mean full-context success: %.3f" % results["mean_full_success"])
    print("")
    print("%-10s %s" % ("strategy",
                         "  ".join("%4d%%" % int(b * 100)
                                   for b in BUDGETS)))
    for strategy in STRATEGIES:
        cells = "  ".join("%.3f" % results["packing"][strategy]["%.2f" % b]
                          for b in BUDGETS)
        print("%-10s %s" % (strategy, cells))
    print("")
    masking = results["masking"]
    print("Redundancy masking: duplicated key prices %.3f per copy, "
          "%.3f jointly" % (masking["mean_individual"],
                            masking["mean_joint"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
