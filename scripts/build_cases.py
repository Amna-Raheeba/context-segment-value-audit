"""Build the shipped demo case: one contradictory_tool task,
serialised with its planted ground truth, at data/demo_task.json.

Run from the repository root:  python scripts/build_cases.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_segment_value.generate import build_task, task_to_dict

DEMO_SCENARIO = "contradictory_tool"
DEMO_SEED = 3


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task = build_task(DEMO_SCENARIO, DEMO_SEED)
    path = os.path.join(root, "data", "demo_task.json")
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(task_to_dict(task), fh, indent=2)
    print("Wrote %s (%s, %d segments)" % (path, task.task_id,
                                         len(task.segments)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
