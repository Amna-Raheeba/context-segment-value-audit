"""Causal segment pricing by leave-one-out replay.

A segment's *value* is the change in measured task success when the
segment is removed from the full context and the task is replayed
under the same draws:

    value(s) = success(full) - success(full without s)

Positive: the segment earns its tokens. Near zero: it is inert
ballast. Negative: the context is better off without it -- the
segment is a liability, and it goes on the negative-value register.

Value density divides value by token count, which is the number a
budget packer actually optimises.

Leave-one-out pricing has one known blind spot, and this module
measures it instead of hiding it: redundancy masking. Two copies of
a key fact each price near zero alone (remove one, the other still
carries the task), while removing both is expensive. The duplicate
probe finds near-duplicate pairs by text similarity and prices them
jointly, so the register can say "individually cheap, jointly
load-bearing".
"""

from .solver import Solver


def measure_values(task, solver=None):
    """Leave-one-out value of every segment, from sampled replays."""
    if solver is None:
        solver = Solver()
    full_ids = task.segment_ids()
    full_success = solver.success_rate(task, full_ids)
    values = {}
    for seg in task.segments:
        reduced = [sid for sid in full_ids if sid != seg.segment_id]
        values[seg.segment_id] = full_success - solver.success_rate(
            task, reduced)
    return values


def value_density(value, tokens):
    if tokens <= 0:
        return 0.0
    return value / float(tokens)


def token_set(text):
    return set(text.lower().split())


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / float(len(union))


def find_duplicate_pairs(task, threshold=0.45):
    """Near-duplicate segment pairs by Jaccard similarity of their
    token sets. Returns (id_a, id_b, similarity) sorted by
    similarity, strongest first."""
    pairs = []
    segments = task.segments
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            sim = jaccard(token_set(segments[i].text),
                          token_set(segments[j].text))
            if sim >= threshold:
                pairs.append((segments[i].segment_id,
                              segments[j].segment_id, sim))
    pairs.sort(key=lambda item: -item[2])
    return pairs


def joint_value(task, segment_ids, solver=None):
    """Value of a *set* of segments removed together: how much
    success the full context loses when all of them leave at once.
    For a redundant pair this is large even when each member's
    leave-one-out value is near zero."""
    if solver is None:
        solver = Solver()
    removed = set(segment_ids)
    full_ids = task.segment_ids()
    reduced = [sid for sid in full_ids if sid not in removed]
    return solver.success_rate(task, full_ids) - solver.success_rate(
        task, reduced)
