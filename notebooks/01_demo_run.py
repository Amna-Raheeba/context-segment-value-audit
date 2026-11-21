"""Notebook 01 -- demo run.

Generate one task from each scenario family, audit it, and print
the segment prices with the negative-value register. Runnable as a
plain script:  python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_segment_value.audit import audit_task
from context_segment_value.generate import SCENARIOS, build_task
from context_segment_value.report import format_audit_report


def main():
    for scenario in ("contradictory_tool", "redundant_key"):
        task = build_task(scenario, 3 if scenario == "contradictory_tool"
                          else 2)
        print(format_audit_report(audit_task(task)))
    print("Scenario families available: %s" % ", ".join(SCENARIOS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
