"""Command line: demo audit of one task, or the full evaluation."""

import argparse
import os
import sys

from .audit import audit_task
from .evaluate import run_evaluation, write_evaluation
from .generate import SCENARIOS, build_task
from .report import format_audit_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="context_segment_value",
        description="Price context segments by their causal "
                    "leave-one-out effect on task success.")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="audit one generated task")
    demo.add_argument("--scenario", default="contradictory_tool",
                      choices=list(SCENARIOS))
    demo.add_argument("--seed", type=int, default=3)

    evaluate = sub.add_parser("evaluate", help="run the seeded "
                                               "evaluation")
    evaluate.add_argument("--out", default="results",
                          help="output directory")
    evaluate.add_argument("--seeds", type=int, default=8,
                          help="seeds per scenario family (0..n-1)")

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        results = run_evaluation(seeds=tuple(range(args.seeds)))
        paths = write_evaluation(results, args.out)
        print("Wrote evaluation for %d tasks to %s" % (
            results["n_tasks"], os.path.abspath(args.out)))
        for key in sorted(paths):
            print("  %s: %s" % (key, paths[key]))
        det = results["detection"]
        print("Negative-value detection: P %.3f R %.3f F1 %.3f" % (
            det["precision"], det["recall"], det["f1"]))
        print("Success retained at 50%% budget (value packing): "
              "%.3f" % results["packing"]["value"]["retained_at_50"])
        return 0

    # Default: demo.
    scenario = getattr(args, "scenario", "contradictory_tool")
    seed = getattr(args, "seed", 3)
    task = build_task(scenario, seed)
    audit = audit_task(task)
    sys.stdout.write(format_audit_report(audit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
