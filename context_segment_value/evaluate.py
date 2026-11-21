"""Seeded evaluation against planted ground truth.

Suite: six scenario families x seeds 0..7 = 48 tasks. Per task the
evaluation measures:

1. **Negative-value detection** -- a segment counts as predicted
   negative when its verdict is ``liability``; ground truth is the
   planted class (distractor / stale / contradictory). Reported as
   precision / recall / F1 overall and recall per negative class.
2. **Value accuracy** -- mean absolute error of the measured
   leave-one-out values against the solver's analytic marginals.
3. **Redundancy masking** -- in ``redundant_key`` tasks, the mean
   individual price of the two key copies vs their joint price.
4. **Budget packing** -- mean measured success per strategy
   (value / relevance / recency / oracle) per budget fraction, plus
   each task's full-context success. Headline: success retained at
   the 50% budget, per strategy.
5. **Replay-budget sensitivity** -- detection re-scored when the
   pricing gets only 25 / 50 / 100 of its 400 paired replays, so
   the cost of a cheaper audit is measured, not guessed.
"""

import io
import json
import os

from .audit import audit_task
from .generate import SCENARIOS, build_suite
from .models import Task
from .packing import BUDGETS, STRATEGIES, pack_and_measure
from .solver import Solver
from .value import joint_value, measure_values

DEFAULT_SEEDS = tuple(range(8))

# Replay budgets for the detection-sensitivity measurement: how
# many paired solver draws the leave-one-out pricing gets.
DRAW_GRID = (25, 50, 100, 400)


def _detection_counts(tasks, solver):
    tp = fp = fn = 0
    per_class = {}
    for task in tasks:
        audit = audit_task(task, solver)
        predicted = set(row.segment.segment_id
                        for row in audit.liabilities)
        for seg in task.segments:
            truth_negative = seg.is_negative
            if seg.value_class not in per_class:
                per_class[seg.value_class] = {"tp": 0, "fn": 0, "n": 0}
            entry = per_class[seg.value_class]
            if truth_negative:
                entry["n"] += 1
                if seg.segment_id in predicted:
                    tp += 1
                    entry["tp"] += 1
                else:
                    fn += 1
                    entry["fn"] += 1
            elif seg.segment_id in predicted:
                fp += 1
    precision = tp / float(tp + fp) if (tp + fp) else 1.0
    recall = tp / float(tp + fn) if (tp + fn) else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    per_class_recall = {}
    for value_class in sorted(per_class):
        entry = per_class[value_class]
        if entry["n"]:
            per_class_recall[value_class] = entry["tp"] / float(entry["n"])
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "per_class_recall": per_class_recall,
    }


def _with_draws(task, n_draws):
    """A clone of the task scored on only the first n_draws of its
    paired draws -- the same measurement, with a smaller replay
    budget."""
    clone = Task(task.task_id, task.scenario, task.seed, task.query,
                 task.segments)
    clone.draws = task.draws[:n_draws]
    return clone


def _detection_by_draws(tasks, solver):
    """Negative-value detection as a function of the replay budget.
    Coarse replays quantise success rates, and the audit's failure
    mode there is false alarms on inert segments -- measured here
    instead of asserted."""
    by_draws = {}
    for n_draws in DRAW_GRID:
        clones = [_with_draws(task, n_draws) for task in tasks]
        det = _detection_counts(clones, solver)
        by_draws[str(n_draws)] = {
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "fp": det["fp"],
            "fn": det["fn"],
        }
    return by_draws


def _value_errors(tasks, solver):
    errors = []
    for task in tasks:
        measured = measure_values(task, solver)
        for seg in task.segments:
            analytic = solver.analytic_marginal(task, seg.segment_id)
            errors.append(abs(measured[seg.segment_id] - analytic))
    if not errors:
        return 0.0
    return sum(errors) / float(len(errors))


def _masking(tasks, solver):
    individual = []
    joint = []
    for task in tasks:
        if task.scenario != "redundant_key":
            continue
        measured = measure_values(task, solver)
        copies = [seg for seg in task.segments
                  if seg.dup_group is not None]
        if len(copies) < 2:
            continue
        for seg in copies:
            individual.append(measured[seg.segment_id])
        joint.append(joint_value(
            task, [seg.segment_id for seg in copies], solver))
    if not individual or not joint:
        return {"mean_individual": 0.0, "mean_joint": 0.0, "tasks": 0}
    return {
        "mean_individual": sum(individual) / float(len(individual)),
        "mean_joint": sum(joint) / float(len(joint)),
        "tasks": len(joint),
    }


def _packing(tasks, solver):
    per_strategy = {}
    for strategy in STRATEGIES:
        per_strategy[strategy] = {}
        for budget in BUDGETS:
            per_strategy[strategy]["%.2f" % budget] = []
    full_success = []
    rows = []
    for task in tasks:
        values = measure_values(task, solver)
        full = solver.success_rate(task, task.segment_ids())
        full_success.append(full)
        for strategy in STRATEGIES:
            for budget in BUDGETS:
                packed, tokens, success = pack_and_measure(
                    task, strategy, budget, values=values, solver=solver)
                per_strategy[strategy]["%.2f" % budget].append(success)
                rows.append({
                    "task_id": task.task_id,
                    "scenario": task.scenario,
                    "strategy": strategy,
                    "budget": budget,
                    "tokens_used": tokens,
                    "success": success,
                    "full_success": full,
                })
    mean_full = sum(full_success) / float(len(full_success))
    aggregate = {}
    for strategy in STRATEGIES:
        aggregate[strategy] = {}
        for budget in BUDGETS:
            key = "%.2f" % budget
            vals = per_strategy[strategy][key]
            aggregate[strategy][key] = sum(vals) / float(len(vals))
        # Retention is a ratio of aggregate means, not a mean of
        # per-task ratios: tasks whose full context nearly always
        # fails would otherwise dominate the average. A ratio above
        # 1.0 is real and means the budget cut *helped* -- the
        # packer evicted liabilities the full context keeps.
        aggregate[strategy]["retained_at_50"] = (
            aggregate[strategy]["0.50"] / mean_full if mean_full else 0.0)
    return aggregate, mean_full, rows


def run_evaluation(seeds=DEFAULT_SEEDS):
    solver = Solver()
    tasks = build_suite(seeds)
    detection = _detection_counts(tasks, solver)
    detection_by_draws = _detection_by_draws(tasks, solver)
    value_mae = _value_errors(tasks, solver)
    masking = _masking(tasks, solver)
    packing_agg, mean_full, packing_rows = _packing(tasks, solver)
    per_scenario = {}
    for scenario in SCENARIOS:
        scen_rows = [row for row in packing_rows
                     if row["scenario"] == scenario and row["budget"] == 0.50]
        entry = {}
        for strategy in STRATEGIES:
            vals = [row["success"] for row in scen_rows
                    if row["strategy"] == strategy]
            if vals:
                entry[strategy] = sum(vals) / float(len(vals))
        fulls = [row["full_success"] for row in scen_rows]
        entry["full"] = sum(fulls) / float(len(fulls)) if fulls else 0.0
        per_scenario[scenario] = entry
    return {
        "n_tasks": len(tasks),
        "scenarios": list(SCENARIOS),
        "seeds": list(seeds),
        "detection": detection,
        "detection_by_draws": detection_by_draws,
        "value_mae": value_mae,
        "masking": masking,
        "packing": packing_agg,
        "mean_full_success": mean_full,
        "per_scenario_at_50": per_scenario,
        "packing_rows": packing_rows,
    }


def write_evaluation(results, out_dir):
    """Write evaluation.json / evaluation.csv /
    evaluation_summary.md / HEADLINE.md into out_dir."""
    from .report import evaluation_summary_markdown, headline_markdown

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    json_path = os.path.join(out_dir, "evaluation.json")
    with io.open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    csv_path = os.path.join(out_dir, "evaluation.csv")
    with io.open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("task_id,scenario,strategy,budget,tokens_used,"
                 "success,full_success\n")
        for row in results["packing_rows"]:
            fh.write("%s,%s,%s,%.2f,%d,%.4f,%.4f\n" % (
                row["task_id"], row["scenario"], row["strategy"],
                row["budget"], row["tokens_used"], row["success"],
                row["full_success"]))
    summary_path = os.path.join(out_dir, "evaluation_summary.md")
    with io.open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(evaluation_summary_markdown(results))
    headline_path = os.path.join(out_dir, "HEADLINE.md")
    with io.open(headline_path, "w", encoding="utf-8") as fh:
        fh.write(headline_markdown(results))
    return {
        "json": json_path, "csv": csv_path,
        "summary": summary_path, "headline": headline_path,
    }
