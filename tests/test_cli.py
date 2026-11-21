"""Tests for the command line."""

import os

from context_segment_value.cli import main


def test_demo_default(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Negative-value register" in out
    assert "contradictory" in out


def test_demo_redundant_shows_masking(capsys):
    assert main(["demo", "--scenario", "redundant_key",
                 "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert "REDUNDANCY MASKING" in out


def test_evaluate_writes_outputs(tmp_path):
    assert main(["evaluate", "--out", str(tmp_path),
                 "--seeds", "2"]) == 0
    for name in ("evaluation.json", "evaluation.csv",
                 "evaluation_summary.md", "HEADLINE.md"):
        assert os.path.exists(os.path.join(str(tmp_path), name))
