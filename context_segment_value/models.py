"""Data model: context segments, tasks, and planted ground truth.

A *task* is one agent job: a query plus the context segments the
agent would be given. Every segment carries a planted *value class*
and a planted *contribution* -- the ground truth the audit is scored
against. The contribution is what the segment does to the solver's
success logit when it is present in the context:

- ``key``           strongly positive -- carries the fact the task
                    turns on;
- ``supporting``    moderately positive -- partial, corroborating
                    information;
- ``neutral``       ~zero -- history chatter, filler;
- ``distractor``    negative -- topically similar to the query but
                    wrong, so it pulls the solver off the answer;
- ``stale``         negative -- an outdated instruction that
                    conflicts with the current one;
- ``contradictory`` negative -- a tool output that contradicts the
                    key fact with apparent authority;
- ``duplicate_key`` positive, but a near-copy of a ``key`` segment;
                    the solver discounts repeat copies, which is
                    what makes marginal pricing mask redundancy.

Roles describe where a segment sits in a real prompt (system text,
instruction, retrieved chunk, tool output, history turn); the value
class describes what it *does*. The two are deliberately independent:
a tool output can be the key fact or the contradiction.
"""

import random

SEGMENT_ROLES = ("system", "instruction", "retrieved", "tool_output", "history")

VALUE_CLASSES = (
    "key",
    "supporting",
    "neutral",
    "distractor",
    "stale",
    "contradictory",
    "duplicate_key",
)

NEGATIVE_CLASSES = ("distractor", "stale", "contradictory")

POSITIVE_CLASSES = ("key", "supporting", "duplicate_key")

# Planted contribution of each value class, in success-logit units.
# The generator jitters neutral segments slightly; the rest are exact
# so the analytic ground truth stays auditable.
BASE_CONTRIBUTIONS = {
    "key": 2.2,
    "supporting": 0.9,
    "neutral": 0.0,
    "distractor": -1.3,
    "stale": -1.7,
    "contradictory": -1.9,
    "duplicate_key": 2.2,
}

# Number of solver draws per task. Success is measured by replaying
# the task this many times under common random numbers, so two
# contexts for the same task face the *same* draws and their
# difference is a paired comparison, not sampling noise.
N_DRAWS = 400


def make_draws(seed, n=N_DRAWS):
    """Deterministic uniform draws for one task's solver replays."""
    rng = random.Random(seed * 1000003 + 77)
    return tuple(rng.random() for _ in range(n))


class Segment(object):
    """One priced unit of context."""

    def __init__(self, segment_id, role, value_class, text, position,
                 contribution, dup_group=None):
        if role not in SEGMENT_ROLES:
            raise ValueError("unknown role: %s" % role)
        if value_class not in VALUE_CLASSES:
            raise ValueError("unknown value class: %s" % value_class)
        self.segment_id = segment_id
        self.role = role
        self.value_class = value_class
        self.text = text
        self.position = position
        self.contribution = contribution
        # Segments sharing a dup_group are near-copies of one fact;
        # the solver discounts every copy after the first.
        self.dup_group = dup_group

    @property
    def tokens(self):
        return len(self.text.split())

    @property
    def is_negative(self):
        return self.value_class in NEGATIVE_CLASSES

    def __repr__(self):
        return "Segment(%s, %s, %s)" % (
            self.segment_id, self.role, self.value_class)


class Task(object):
    """One agent job: a query, its context segments, and the draws
    its solver replays are scored against."""

    def __init__(self, task_id, scenario, seed, query, segments):
        self.task_id = task_id
        self.scenario = scenario
        self.seed = seed
        self.query = query
        self.segments = list(segments)
        self.draws = make_draws(seed)
        self._by_id = {}
        for seg in self.segments:
            if seg.segment_id in self._by_id:
                raise ValueError("duplicate segment id: %s" % seg.segment_id)
            self._by_id[seg.segment_id] = seg

    def segment(self, segment_id):
        return self._by_id[segment_id]

    def segment_ids(self):
        return [seg.segment_id for seg in self.segments]

    @property
    def total_tokens(self):
        return sum(seg.tokens for seg in self.segments)

    def negative_segments(self):
        return [seg for seg in self.segments if seg.is_negative]

    def __repr__(self):
        return "Task(%s, %s, %d segments)" % (
            self.task_id, self.scenario, len(self.segments))
