"""Tests for the seeded task generator."""

import pytest

from context_segment_value.generate import (SCENARIOS, build_suite,
                                            build_task, task_from_dict,
                                            task_to_dict)

EXPECTED_COUNTS = {
    "standard_mix": 6,
    "stale_instruction": 6,
    "contradictory_tool": 6,
    "redundant_key": 6,
    "bloated_history": 10,
    "distractor_heavy": 7,
}

EXPECTED_NEGATIVES = {
    "standard_mix": 1,
    "stale_instruction": 2,
    "contradictory_tool": 1,
    "redundant_key": 1,
    "bloated_history": 1,
    "distractor_heavy": 3,
}


def test_build_task_is_deterministic():
    task_a = build_task("standard_mix", 4)
    task_b = build_task("standard_mix", 4)
    assert [seg.text for seg in task_a.segments] == \
        [seg.text for seg in task_b.segments]
    assert [seg.contribution for seg in task_a.segments] == \
        [seg.contribution for seg in task_b.segments]


def test_different_seeds_give_different_tasks():
    task_a = build_task("standard_mix", 0)
    task_b = build_task("standard_mix", 1)
    assert task_a.query != task_b.query


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenario_segment_counts(scenario):
    task = build_task(scenario, 0)
    assert len(task.segments) == EXPECTED_COUNTS[scenario]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenario_planted_negatives(scenario):
    task = build_task(scenario, 0)
    assert len(task.negative_segments()) == EXPECTED_NEGATIVES[scenario]


def test_planted_contributions_by_class():
    task = build_task("standard_mix", 0)
    by_class = {}
    for seg in task.segments:
        by_class.setdefault(seg.value_class, []).append(seg.contribution)
    assert by_class["key"] == [2.2]
    assert by_class["supporting"] == [0.9]
    assert by_class["distractor"] == [-1.3]
    for contribution in by_class["neutral"]:
        assert -0.05 <= contribution <= 0.05


def test_positions_are_prompt_order():
    task = build_task("bloated_history", 2)
    assert [seg.position for seg in task.segments] == \
        list(range(len(task.segments)))


def test_redundant_key_shares_dup_group():
    task = build_task("redundant_key", 1)
    grouped = [seg for seg in task.segments if seg.dup_group is not None]
    assert len(grouped) == 2
    assert grouped[0].dup_group == grouped[1].dup_group
    assert set(seg.value_class for seg in grouped) == \
        {"key", "duplicate_key"}


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        build_task("no_such_scenario", 0)


def test_build_suite_size():
    assert len(build_suite()) == 48
    assert len(build_suite(seeds=(0, 1))) == 12


def test_serialisation_round_trip():
    task = build_task("contradictory_tool", 3)
    clone = task_from_dict(task_to_dict(task))
    assert clone.task_id == task.task_id
    assert clone.query == task.query
    assert [seg.text for seg in clone.segments] == \
        [seg.text for seg in task.segments]
    assert [seg.contribution for seg in clone.segments] == \
        [seg.contribution for seg in task.segments]
    assert len(clone.draws) == len(task.draws)
