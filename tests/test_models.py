"""Tests for the data model."""

import pytest

from context_segment_value.models import (NEGATIVE_CLASSES, Segment,
                                          Task, make_draws)


def _seg(seg_id="s01", role="retrieved", value_class="key",
         text="alpha beta gamma", position=0, contribution=2.2):
    return Segment(seg_id, role, value_class, text, position, contribution)


def test_segment_tokens_counts_words():
    assert _seg(text="one two three four").tokens == 4


def test_segment_rejects_unknown_role():
    with pytest.raises(ValueError):
        _seg(role="memo")


def test_segment_rejects_unknown_value_class():
    with pytest.raises(ValueError):
        _seg(value_class="helpful")


def test_segment_is_negative_matches_classes():
    for value_class in NEGATIVE_CLASSES:
        assert _seg(value_class=value_class).is_negative
    assert not _seg(value_class="key").is_negative
    assert not _seg(value_class="neutral").is_negative


def test_make_draws_deterministic_and_bounded():
    draws_a = make_draws(5)
    draws_b = make_draws(5)
    assert draws_a == draws_b
    assert len(draws_a) == 400
    assert all(0.0 <= draw < 1.0 for draw in draws_a)
    assert make_draws(6) != draws_a


def test_task_rejects_duplicate_segment_ids():
    with pytest.raises(ValueError):
        Task("t", "standard_mix", 0, "q", [_seg("s01"), _seg("s01")])


def test_task_aggregates():
    segments = [
        _seg("s01", text="a b c"),
        _seg("s02", value_class="distractor", text="a b",
             contribution=-1.3),
    ]
    task = Task("t", "standard_mix", 0, "q", segments)
    assert task.total_tokens == 5
    assert task.segment_ids() == ["s01", "s02"]
    assert task.segment("s02").value_class == "distractor"
    assert [seg.segment_id for seg in task.negative_segments()] == ["s02"]
    assert len(task.draws) == 400
