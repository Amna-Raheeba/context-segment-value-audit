"""Budget packing: which segments survive a token budget, decided
four ways.

- ``value``     rank by measured leave-one-out value density and
                never pack a segment whose measured value is
                negative -- under any budget, a liability is left
                out on purpose;
- ``relevance`` rank by TF-IDF cosine to the query (the standard
                baseline, and the one distractors are built to
                fool);
- ``recency``   rank by prompt position, latest first -- the other
                common heuristic ("keep the recent turns");
- ``oracle``    rank by *true* analytic value density. Not a usable
                strategy -- it peeks at ground truth -- but the
                ceiling the measured-value packer is compared to.

Packing is greedy in rank order: a segment is added if it still fits
the remaining budget, otherwise it is skipped and the next is tried.
"""

from .relevance import TfidfScorer
from .solver import Solver
from .value import measure_values, value_density

STRATEGIES = ("value", "relevance", "recency", "oracle")

BUDGETS = (0.25, 0.50, 0.75, 1.00)


def rank_segments(task, strategy, values=None, solver=None):
    """Segment ids in packing order for one strategy."""
    if solver is None:
        solver = Solver()
    segments = task.segments
    if strategy == "value":
        if values is None:
            values = measure_values(task, solver)
        return sorted(
            (seg.segment_id for seg in segments),
            key=lambda sid: -value_density(
                values[sid], task.segment(sid).tokens))
    if strategy == "relevance":
        scorer = TfidfScorer(task)
        return sorted(
            (seg.segment_id for seg in segments),
            key=lambda sid: -scorer.score(sid))
    if strategy == "recency":
        return [seg.segment_id for seg in
                sorted(segments, key=lambda seg: -seg.position)]
    if strategy == "oracle":
        return sorted(
            (seg.segment_id for seg in segments),
            key=lambda sid: -value_density(
                solver.analytic_marginal(task, sid),
                task.segment(sid).tokens))
    raise ValueError("unknown strategy: %s" % strategy)


def _excluded_negatives(task, strategy, values, solver):
    """Segments a strategy refuses on principle: value and oracle
    packing never include a segment priced negative."""
    if strategy == "value":
        if values is None:
            values = measure_values(task, solver)
        return set(sid for sid, val in values.items() if val < 0.0)
    if strategy == "oracle":
        return set(
            seg.segment_id for seg in task.segments
            if solver.analytic_marginal(task, seg.segment_id) < 0.0)
    return set()


def pack(task, strategy, budget_fraction, values=None, solver=None):
    """Greedy-fill a token budget in the strategy's rank order.
    Returns the packed segment ids (in rank order)."""
    if solver is None:
        solver = Solver()
    budget_tokens = int(round(task.total_tokens * budget_fraction))
    ranking = rank_segments(task, strategy, values=values, solver=solver)
    excluded = _excluded_negatives(task, strategy, values, solver)
    packed = []
    used = 0
    for sid in ranking:
        if sid in excluded:
            continue
        tokens = task.segment(sid).tokens
        if used + tokens <= budget_tokens:
            packed.append(sid)
            used += tokens
    return packed


def pack_and_measure(task, strategy, budget_fraction, values=None,
                     solver=None):
    """Pack under a budget and replay: (packed ids, tokens used,
    measured success)."""
    if solver is None:
        solver = Solver()
    packed = pack(task, strategy, budget_fraction, values=values,
                  solver=solver)
    tokens = sum(task.segment(sid).tokens for sid in packed)
    success = solver.success_rate(task, packed)
    return packed, tokens, success
