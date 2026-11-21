"""Tests for leave-one-out pricing and the duplicate probe."""

from context_segment_value.generate import build_task
from context_segment_value.solver import Solver
from context_segment_value.value import (find_duplicate_pairs, jaccard,
                                         joint_value, measure_values,
                                         value_density)


def test_measure_values_covers_every_segment():
    task = build_task("standard_mix", 0)
    values = measure_values(task)
    assert set(values) == set(task.segment_ids())


def test_measured_values_recover_planted_signs():
    task = build_task("contradictory_tool", 3)
    values = measure_values(task)
    for seg in task.segments:
        if seg.value_class == "key":
            assert values[seg.segment_id] > 0.1
        elif seg.value_class == "contradictory":
            assert values[seg.segment_id] < -0.1


def test_measured_values_track_analytic_marginals():
    task = build_task("stale_instruction", 2)
    solver = Solver()
    values = measure_values(task, solver)
    for seg in task.segments:
        analytic = solver.analytic_marginal(task, seg.segment_id)
        assert abs(values[seg.segment_id] - analytic) < 0.06


def test_value_density_normalises_by_tokens():
    assert value_density(0.5, 25) == 0.02
    assert value_density(-0.4, 8) == -0.05
    assert value_density(0.5, 0) == 0.0


def test_contradiction_has_the_worst_density():
    task = build_task("contradictory_tool", 3)
    values = measure_values(task)
    densities = dict(
        (seg.segment_id, value_density(values[seg.segment_id], seg.tokens))
        for seg in task.segments)
    worst = min(densities, key=lambda sid: densities[sid])
    assert task.segment(worst).value_class == "contradictory"


def test_jaccard_reference_values():
    assert jaccard(set(["a", "b"]), set(["a", "b"])) == 1.0
    assert jaccard(set(["a"]), set(["b"])) == 0.0
    assert jaccard(set(), set()) == 1.0
    assert abs(jaccard(set(["a", "b"]), set(["b", "c"])) - 1.0 / 3.0) < 1e-9


def test_find_duplicate_pairs_on_redundant_task():
    task = build_task("redundant_key", 2)
    pairs = find_duplicate_pairs(task)
    assert len(pairs) >= 1
    id_a, id_b, sim = pairs[0]
    classes = {task.segment(id_a).value_class,
               task.segment(id_b).value_class}
    assert classes == {"key", "duplicate_key"}
    assert sim >= 0.45


def test_find_duplicate_pairs_empty_when_no_copies():
    task = build_task("contradictory_tool", 3)
    assert find_duplicate_pairs(task) == []


def test_joint_value_exposes_masking():
    task = build_task("redundant_key", 2)
    values = measure_values(task)
    copies = [seg.segment_id for seg in task.segments
              if seg.dup_group is not None]
    joint = joint_value(task, copies)
    for seg_id in copies:
        assert joint > values[seg_id] + 0.2


def test_joint_value_of_single_segment_equals_loo():
    task = build_task("standard_mix", 1)
    values = measure_values(task)
    key = [seg for seg in task.segments
           if seg.value_class == "key"][0]
    assert abs(joint_value(task, [key.segment_id])
               - values[key.segment_id]) < 1e-9
