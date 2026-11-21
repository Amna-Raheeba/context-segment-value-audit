"""Tests for the TF-IDF relevance baseline."""

from context_segment_value.generate import build_task
from context_segment_value.relevance import TfidfScorer, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("The Cutover-Window, 02:00!") == \
        ["the", "cutover", "window", "02", "00"]


def test_scores_cover_all_segments_in_range():
    task = build_task("standard_mix", 0)
    scorer = TfidfScorer(task)
    scores = scorer.scores()
    assert set(scores) == set(task.segment_ids())
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_distractors_outrank_history_on_relevance():
    # The trap the baseline falls for: distractors are built from the
    # query's own vocabulary, history chatter shares almost none.
    task = build_task("distractor_heavy", 0)
    scorer = TfidfScorer(task)
    distractor_scores = [scorer.score(seg.segment_id)
                         for seg in task.negative_segments()]
    history_scores = [scorer.score(seg.segment_id) for seg in
                      task.segments if seg.role == "history"]
    assert min(distractor_scores) > max(history_scores)


def test_key_outranks_history_on_relevance():
    task = build_task("standard_mix", 0)
    scorer = TfidfScorer(task)
    key = [seg for seg in task.segments
           if seg.value_class == "key"][0]
    history = [seg for seg in task.segments if seg.role == "history"]
    assert all(scorer.score(key.segment_id) > scorer.score(seg.segment_id)
               for seg in history)


def test_cosine_of_identical_vectors_is_one():
    vector = {"alpha": 1.5, "beta": 0.5}
    assert abs(TfidfScorer._cosine(vector, vector) - 1.0) < 1e-9
    assert TfidfScorer._cosine({}, vector) == 0.0
