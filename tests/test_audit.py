"""Tests for the audit and its verdicts."""

from context_segment_value.audit import audit_task, verdict_for
from context_segment_value.generate import build_suite, build_task


def test_verdict_thresholds():
    assert verdict_for(0.02) == "asset"
    assert verdict_for(0.5) == "asset"
    assert verdict_for(-0.02) == "liability"
    assert verdict_for(-0.5) == "liability"
    assert verdict_for(0.0) == "inert"
    assert verdict_for(0.019) == "inert"
    assert verdict_for(-0.019) == "inert"


def test_audit_flags_the_contradiction():
    task = build_task("contradictory_tool", 3)
    audit = audit_task(task)
    liability_ids = [row.segment.segment_id for row in audit.liabilities]
    contradictory = [seg for seg in task.segments
                     if seg.value_class == "contradictory"][0]
    assert liability_ids == [contradictory.segment_id]
    key_row = audit.row_for(
        [seg for seg in task.segments
         if seg.value_class == "key"][0].segment_id)
    assert key_row.verdict == "asset"


def test_audit_rows_sorted_by_value():
    task = build_task("standard_mix", 0)
    audit = audit_task(task)
    values = [row.value for row in audit.rows]
    assert values == sorted(values, reverse=True)


def test_audit_detects_masking_in_redundant_task():
    task = build_task("redundant_key", 2)
    audit = audit_task(task)
    assert len(audit.probes) >= 1
    assert audit.probes[0].masked


def test_audit_has_no_probes_without_duplicates():
    task = build_task("contradictory_tool", 3)
    audit = audit_task(task)
    assert audit.probes == []


def test_audit_is_deterministic():
    task = build_task("stale_instruction", 1)
    first = audit_task(task)
    second = audit_task(task)
    assert [row.value for row in first.rows] == \
        [row.value for row in second.rows]


def test_suite_wide_detection_is_exact_at_full_replays():
    # The evaluation's headline claim, restated as a test: at 400
    # paired replays every planted negative in the suite is flagged
    # and nothing else is.
    for task in build_suite():
        audit = audit_task(task)
        predicted = set(row.segment.segment_id
                        for row in audit.liabilities)
        truth = set(seg.segment_id for seg in task.negative_segments())
        assert predicted == truth, task.task_id
