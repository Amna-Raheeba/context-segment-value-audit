"""Tests for the seeded evaluation."""

import io
import json
import os

from context_segment_value.evaluate import run_evaluation, write_evaluation
from context_segment_value.packing import STRATEGIES


def _small_results():
    return run_evaluation(seeds=(0, 1))


def test_evaluation_structure():
    results = _small_results()
    assert results["n_tasks"] == 12
    for key in ("detection", "detection_by_draws", "value_mae",
                "masking", "packing", "mean_full_success",
                "per_scenario_at_50", "packing_rows"):
        assert key in results
    for strategy in STRATEGIES:
        assert strategy in results["packing"]
        assert results["packing"][strategy]["retained_at_50"] > 0.0


def test_value_packing_beats_recency_and_full_context():
    results = _small_results()
    packing = results["packing"]
    assert packing["value"]["0.50"] > packing["recency"]["0.50"]
    assert packing["value"]["0.50"] > results["mean_full_success"]
    assert packing["value"]["retained_at_50"] > \
        packing["recency"]["retained_at_50"]


def test_detection_by_draws_shows_replay_budget_effect():
    results = _small_results()
    by_draws = results["detection_by_draws"]
    assert set(by_draws) == {"25", "50", "100", "400"}
    # More replays never hurt precision on this suite, and recall
    # for planted liabilities holds at every budget.
    assert by_draws["400"]["precision"] >= by_draws["50"]["precision"]
    for entry in by_draws.values():
        assert entry["recall"] == 1.0


def test_masking_measured_on_redundant_tasks():
    results = _small_results()
    masking = results["masking"]
    assert masking["tasks"] == 2
    assert masking["mean_joint"] > masking["mean_individual"] + 0.2


def test_write_evaluation_outputs(tmp_path):
    results = _small_results()
    paths = write_evaluation(results, str(tmp_path))
    for key in ("json", "csv", "summary", "headline"):
        assert os.path.exists(paths[key])
    with io.open(paths["json"], encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["n_tasks"] == 12
    with io.open(paths["csv"], encoding="utf-8") as fh:
        lines = fh.read().strip().split("\n")
    # header + 12 tasks x 4 strategies x 4 budgets
    assert len(lines) == 1 + 12 * 4 * 4
    with io.open(paths["headline"], encoding="utf-8") as fh:
        assert "Headline results" in fh.read()
