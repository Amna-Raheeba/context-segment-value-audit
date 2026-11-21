"""Tests for the seeded solver."""

from context_segment_value.generate import build_task
from context_segment_value.models import Segment, Task
from context_segment_value.solver import Solver, sigmoid


def test_sigmoid_reference_values():
    assert sigmoid(0.0) == 0.5
    assert 0.0 < sigmoid(-10.0) < 0.001
    assert 0.999 < sigmoid(10.0) < 1.0


def test_empty_context_probability_is_base_rate():
    task = build_task("standard_mix", 0)
    solver = Solver()
    assert abs(solver.probability(task, []) - sigmoid(-1.2)) < 1e-12


def test_key_raises_and_distractor_lowers_probability():
    task = build_task("standard_mix", 0)
    solver = Solver()
    key = [seg for seg in task.segments if seg.value_class == "key"][0]
    distractor = task.negative_segments()[0]
    empty = solver.probability(task, [])
    assert solver.probability(task, [key.segment_id]) > empty
    assert solver.probability(task, [distractor.segment_id]) < empty


def test_success_rate_bounds_and_determinism():
    task = build_task("distractor_heavy", 1)
    solver = Solver()
    rate_a = solver.success_rate(task, task.segment_ids())
    rate_b = solver.success_rate(task, task.segment_ids())
    assert rate_a == rate_b
    assert 0.0 <= rate_a <= 1.0


def test_full_context_beats_empty_on_standard_task():
    task = build_task("standard_mix", 0)
    solver = Solver()
    assert solver.success_rate(task, task.segment_ids()) > \
        solver.success_rate(task, [])


def _dup_task():
    segments = [
        Segment("s01", "retrieved", "key", "the fact in full", 0, 2.2,
                dup_group="g"),
        Segment("s02", "retrieved", "duplicate_key", "the fact again",
                1, 2.2, dup_group="g"),
    ]
    return Task("t-dup", "redundant_key", 0, "q", segments)


def test_duplicate_discount_counts_second_copy_less():
    task = _dup_task()
    discounted = Solver(duplicate_discount=0.15)
    full_credit = Solver(duplicate_discount=1.0)
    both = task.segment_ids()
    assert discounted.probability(task, both) < \
        full_credit.probability(task, both)


def test_duplicate_marginals_are_small_and_symmetric():
    task = _dup_task()
    solver = Solver()
    marginal_a = solver.analytic_marginal(task, "s01")
    marginal_b = solver.analytic_marginal(task, "s02")
    assert 0.0 < marginal_a < 0.1
    assert abs(marginal_a - marginal_b) < 1e-9
    joint = solver.probability(task, task.segment_ids()) - \
        solver.probability(task, [])
    assert joint > 0.4


def test_analytic_marginal_signs():
    task = build_task("contradictory_tool", 3)
    solver = Solver()
    for seg in task.segments:
        marginal = solver.analytic_marginal(task, seg.segment_id)
        if seg.value_class == "key":
            assert marginal > 0.1
        elif seg.value_class == "contradictory":
            assert marginal < -0.1
