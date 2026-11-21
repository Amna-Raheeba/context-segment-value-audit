"""The audit: price one task's context and register its liabilities.

For each segment the audit reports the measured leave-one-out value,
its value density per token, and a verdict:

- ``asset``     value >= +0.02 -- earns its tokens;
- ``inert``     between -0.02 and +0.02 -- ballast: harmless, but
                the first thing a budget should cut;
- ``liability`` value <= -0.02 -- the context succeeds *more*
                without it. Liabilities form the negative-value
                register, annotated with the planted-class-agnostic
                reason the audit can see (it reports the measured
                harm, not a guessed motive).

The audit also runs the duplicate probe: near-duplicate pairs are
priced jointly so redundancy masking is visible in the report
instead of silently underpricing both copies.
"""

from .solver import Solver
from .value import (find_duplicate_pairs, joint_value, measure_values,
                    value_density)

ASSET_THRESHOLD = 0.02
LIABILITY_THRESHOLD = -0.02


def verdict_for(value):
    if value >= ASSET_THRESHOLD:
        return "asset"
    if value <= LIABILITY_THRESHOLD:
        return "liability"
    return "inert"


class SegmentRow(object):
    def __init__(self, segment, value):
        self.segment = segment
        self.value = value
        self.density = value_density(value, segment.tokens)
        self.verdict = verdict_for(value)


class DuplicateProbe(object):
    def __init__(self, id_a, id_b, similarity, joint, value_a, value_b):
        self.id_a = id_a
        self.id_b = id_b
        self.similarity = similarity
        self.joint_value = joint
        self.value_a = value_a
        self.value_b = value_b

    @property
    def masked(self):
        """True when the pair is jointly worth far more than the sum
        of its members' individual prices -- the masking signature."""
        return self.joint_value > (self.value_a + self.value_b) + 0.05


class AuditResult(object):
    def __init__(self, task, rows, probes, full_success):
        self.task = task
        self.rows = rows
        self.probes = probes
        self.full_success = full_success

    @property
    def liabilities(self):
        return [row for row in self.rows if row.verdict == "liability"]

    @property
    def assets(self):
        return [row for row in self.rows if row.verdict == "asset"]

    def row_for(self, segment_id):
        for row in self.rows:
            if row.segment.segment_id == segment_id:
                return row
        raise KeyError(segment_id)


def audit_task(task, solver=None):
    if solver is None:
        solver = Solver()
    values = measure_values(task, solver)
    rows = [SegmentRow(seg, values[seg.segment_id]) for seg in task.segments]
    rows.sort(key=lambda row: -row.value)
    probes = []
    for id_a, id_b, sim in find_duplicate_pairs(task):
        probes.append(DuplicateProbe(
            id_a, id_b, sim,
            joint_value(task, (id_a, id_b), solver),
            values[id_a], values[id_b]))
    full_success = solver.success_rate(task, task.segment_ids())
    return AuditResult(task, rows, probes, full_success)
