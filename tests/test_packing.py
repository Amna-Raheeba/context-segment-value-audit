"""Tests for budget packing."""

import pytest

from context_segment_value.generate import build_task
from context_segment_value.packing import (BUDGETS, STRATEGIES, pack,
                                           pack_and_measure, rank_segments)
from context_segment_value.solver import Solver


def test_pack_respects_budget_for_all_strategies():
    task = build_task("bloated_history", 3)
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            packed = pack(task, strategy, budget)
            used = sum(task.segment(sid).tokens for sid in packed)
            assert used <= int(round(task.total_tokens * budget))


def test_value_packing_excludes_planted_negatives_at_full_budget():
    task = build_task("distractor_heavy", 0)
    packed = pack(task, "value", 1.00)
    negative_ids = set(seg.segment_id for seg in task.negative_segments())
    assert not (set(packed) & negative_ids)
    assert packed  # the assets still pack


def test_value_packing_at_full_budget_beats_full_context():
    task = build_task("contradictory_tool", 3)
    solver = Solver()
    packed, tokens, success = pack_and_measure(
        task, "value", 1.00, solver=solver)
    full = solver.success_rate(task, task.segment_ids())
    assert success > full
    assert tokens < task.total_tokens


def test_recency_packing_prefers_history_at_small_budget():
    task = build_task("bloated_history", 3)
    packed = pack(task, "recency", 0.25)
    assert packed
    assert all(task.segment(sid).role == "history" for sid in packed)


def test_oracle_ranks_key_first():
    task = build_task("standard_mix", 0)
    ranking = rank_segments(task, "oracle")
    key = [seg for seg in task.segments
           if seg.value_class == "key"][0]
    assert ranking[0] == key.segment_id


def test_relevance_ranks_a_distractor_above_history():
    task = build_task("distractor_heavy", 0)
    ranking = rank_segments(task, "relevance")
    positions = dict((sid, idx) for idx, sid in enumerate(ranking))
    history = [seg for seg in task.segments if seg.role == "history"]
    for seg in task.negative_segments():
        assert all(positions[seg.segment_id] < positions[h.segment_id]
                   for h in history)


def test_unknown_strategy_raises():
    task = build_task("standard_mix", 0)
    with pytest.raises(ValueError):
        pack(task, "shortest_first", 0.5)


def test_pack_and_measure_is_consistent():
    task = build_task("standard_mix", 2)
    packed, tokens, success = pack_and_measure(task, "value", 0.5)
    assert tokens == sum(task.segment(sid).tokens for sid in packed)
    assert 0.0 <= success <= 1.0
