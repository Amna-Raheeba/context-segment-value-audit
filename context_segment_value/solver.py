"""The seeded solver: a synthetic agent whose success probability is
a known function of the segments in its context.

This stands in for a real LLM agent the way the rest of the portfolio
uses seeded synthetic policies: the ground truth (each segment's
contribution) is known by construction, so an audit instrument can
be scored against it instead of merely demonstrated.

Success model: the logit of success is a base rate plus the sum of
the *effective* contributions of the segments present. Effective
contributions apply one realistic twist -- a duplicate discount:
when several segments in the same duplicate group are present, the
first (in prompt order) counts in full and each later copy counts at
``duplicate_discount`` of its planted contribution. A fact retrieved
twice does not help twice.

Measured success replays the task ``len(task.draws)`` times under
common random numbers: attempt *i* succeeds iff draw *i* is below
the context's success probability. Two contexts for the same task
face identical draws, so their measured difference is a paired
comparison with far less noise than independent sampling.
"""

import math


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class Solver(object):
    def __init__(self, base_logit=-1.2, duplicate_discount=0.15):
        self.base_logit = base_logit
        self.duplicate_discount = duplicate_discount

    def effective_logit(self, task, segment_ids):
        """Success logit of a context (a set of segment ids)."""
        wanted = set(segment_ids)
        logit = self.base_logit
        seen_groups = set()
        for seg in task.segments:  # prompt order decides "first copy"
            if seg.segment_id not in wanted:
                continue
            contribution = seg.contribution
            if seg.dup_group is not None:
                if seg.dup_group in seen_groups:
                    contribution = contribution * self.duplicate_discount
                else:
                    seen_groups.add(seg.dup_group)
            logit += contribution
        return logit

    def probability(self, task, segment_ids):
        return sigmoid(self.effective_logit(task, segment_ids))

    def success_rate(self, task, segment_ids):
        """Measured success: share of the task's draws beaten by the
        context's success probability."""
        p = self.probability(task, segment_ids)
        hits = 0
        for draw in task.draws:
            if draw < p:
                hits += 1
        return hits / float(len(task.draws))

    def analytic_marginal(self, task, segment_id):
        """Ground-truth leave-one-out value: the exact change in
        success probability when the segment leaves the full context.
        The audit never sees this; the evaluation scores against it."""
        full = task.segment_ids()
        reduced = [sid for sid in full if sid != segment_id]
        return self.probability(task, full) - self.probability(task, reduced)
