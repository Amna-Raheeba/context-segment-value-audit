"""Tests for report formatting."""

from context_segment_value.audit import audit_task
from context_segment_value.evaluate import run_evaluation
from context_segment_value.generate import build_task
from context_segment_value.report import (evaluation_summary_markdown,
                                          format_audit_report,
                                          headline_markdown)


def test_audit_report_contents():
    task = build_task("contradictory_tool", 3)
    report = format_audit_report(audit_task(task))
    assert task.task_id in report
    assert "Negative-value register" in report
    assert "liability" in report
    for seg in task.segments:
        assert seg.segment_id in report


def test_audit_report_shows_masking():
    task = build_task("redundant_key", 2)
    report = format_audit_report(audit_task(task))
    assert "REDUNDANCY MASKING" in report


def test_headline_and_summary_markdown():
    results = run_evaluation(seeds=(0, 1))
    headline = headline_markdown(results)
    assert "Headline results" in headline
    assert "50%" in headline
    summary = evaluation_summary_markdown(results)
    assert "Budget packing" in summary
    assert "distractor_heavy" in summary
    assert "Replay budget" in summary or "replay" in summary
