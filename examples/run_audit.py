"""Audit the shipped demo task and print its report.

Run from the repository root:  python examples/run_audit.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_segment_value.audit import audit_task
from context_segment_value.generate import task_from_dict
from context_segment_value.report import format_audit_report


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "demo_task.json")
    with io.open(path, encoding="utf-8") as fh:
        task = task_from_dict(json.load(fh))
    audit = audit_task(task)
    sys.stdout.write(format_audit_report(audit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
