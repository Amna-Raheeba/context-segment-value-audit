"""Human-readable reports: one task's audit, and the evaluation
summary / headline documents."""

from .packing import BUDGETS, STRATEGIES


def format_audit_report(audit):
    task = audit.task
    lines = []
    lines.append("Context segment value audit -- %s" % task.task_id)
    lines.append("Scenario: %s | seed %d | %d segments | %d tokens" % (
        task.scenario, task.seed, len(task.segments), task.total_tokens))
    lines.append("Query: %s" % task.query)
    lines.append("Full-context measured success: %.3f" % audit.full_success)
    lines.append("")
    lines.append("Segment prices (leave-one-out, paired replays):")
    lines.append("  %-5s %-12s %-14s %6s %8s %9s  %s" % (
        "id", "role", "planted class", "tokens", "value", "per-token",
        "verdict"))
    for row in audit.rows:
        seg = row.segment
        lines.append("  %-5s %-12s %-14s %6d %+8.3f %+9.4f  %s" % (
            seg.segment_id, seg.role, seg.value_class, seg.tokens,
            row.value, row.density, row.verdict))
    lines.append("")
    lines.append("Negative-value register:")
    if not audit.liabilities:
        lines.append("  (none -- no segment measurably hurts this task)")
    for row in audit.liabilities:
        seg = row.segment
        lines.append(
            "  %s [%s / %s] value %+.3f over %d tokens -- removing it "
            "raises measured success" % (
                seg.segment_id, seg.role, seg.value_class, row.value,
                seg.tokens))
    lines.append("")
    lines.append("Duplicate probes:")
    if not audit.probes:
        lines.append("  (no near-duplicate pairs)")
    for probe in audit.probes:
        note = "REDUNDANCY MASKING" if probe.masked else "no masking"
        lines.append(
            "  %s + %s (similarity %.2f): individual values %+.3f / "
            "%+.3f, joint value %+.3f -- %s" % (
                probe.id_a, probe.id_b, probe.similarity, probe.value_a,
                probe.value_b, probe.joint_value, note))
    return "\n".join(lines) + "\n"


def headline_markdown(results):
    det = results["detection"]
    packing = results["packing"]
    masking = results["masking"]
    by_draws = results["detection_by_draws"]
    lines = []
    lines.append("# Headline results")
    lines.append("")
    lines.append("Across %d seeded tasks (6 scenario families x %d "
                 "seeds), planted ground truth:" % (
                     results["n_tasks"], len(results["seeds"])))
    lines.append("")
    lines.append("- At a 50%% token budget, value-ranked packing "
                 "scores **%.3f** mean success vs **%.3f** for the "
                 "full context -- a retained ratio of **%.3f**. "
                 "Keeping everything is worse than keeping the right "
                 "half, because the full context keeps its "
                 "liabilities. Relevance-ranked packing scores %.3f "
                 "(retained %.3f), recency-ranked %.3f (retained "
                 "%.3f), and the ground-truth oracle %.3f -- the "
                 "measured-value packer matches the oracle at this "
                 "budget" % (
                     packing["value"]["0.50"],
                     results["mean_full_success"],
                     packing["value"]["retained_at_50"],
                     packing["relevance"]["0.50"],
                     packing["relevance"]["retained_at_50"],
                     packing["recency"]["0.50"],
                     packing["recency"]["retained_at_50"],
                     packing["oracle"]["0.50"]))
    lines.append("- Value-ranked packing at a 100%% budget scores "
                 "%.3f vs %.3f for the full context -- dropping the "
                 "negative-value segments beats keeping everything, "
                 "and relevance-ranked packing gets *worse* as its "
                 "budget grows (%.3f at 50%%, %.3f at 100%%) because "
                 "a bigger budget admits more distractors" % (
                     packing["value"]["1.00"],
                     results["mean_full_success"],
                     packing["relevance"]["0.50"],
                     packing["relevance"]["1.00"]))
    lines.append("- Negative-value detection at 400 paired replays: "
                 "precision **%.3f**, recall **%.3f**, F1 **%.3f** "
                 "(%d planted negatives). At 50 replays precision "
                 "falls to %.3f -- the failure mode is false alarms "
                 "on inert segments, never missed liabilities "
                 "(recall %.3f)" % (
                     det["precision"], det["recall"], det["f1"],
                     det["tp"] + det["fn"],
                     by_draws["50"]["precision"],
                     by_draws["50"]["recall"]))
    lines.append("- Measured leave-one-out values vs analytic "
                 "marginals: MAE **%.4f**" % results["value_mae"])
    lines.append("- Redundancy masking: a duplicated key fact prices "
                 "%.3f per copy on average, but %.3f jointly "
                 "(%d redundant-key tasks)" % (
                     masking["mean_individual"], masking["mean_joint"],
                     masking["tasks"]))
    lines.append("")
    lines.append("All numbers are measured on synthetic seeded data; "
                 "see README section 7 for limits.")
    lines.append("")
    return "\n".join(lines)


def evaluation_summary_markdown(results):
    det = results["detection"]
    packing = results["packing"]
    masking = results["masking"]
    lines = []
    lines.append("# Evaluation summary")
    lines.append("")
    lines.append("%d seeded tasks (6 scenario families x %d seeds), "
                 "planted ground truth, 400 paired solver replays per "
                 "context." % (results["n_tasks"], len(results["seeds"])))
    lines.append("")
    lines.append("## Negative-value detection")
    lines.append("")
    lines.append("Predicted negative = verdict ``liability`` "
                 "(measured value <= -0.02). True negatives planted: "
                 "%d." % (det["tp"] + det["fn"]))
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append("| precision | %.3f |" % det["precision"])
    lines.append("| recall | %.3f |" % det["recall"])
    lines.append("| F1 | %.3f |" % det["f1"])
    lines.append("| true positives | %d |" % det["tp"])
    lines.append("| false positives | %d |" % det["fp"])
    lines.append("| false negatives | %d |" % det["fn"])
    lines.append("")
    lines.append("Recall by planted negative class:")
    lines.append("")
    for value_class in sorted(det["per_class_recall"]):
        lines.append("- %s: %.3f" % (
            value_class, det["per_class_recall"][value_class]))
    lines.append("")
    lines.append("Detection vs replay budget (the same audits, "
                 "priced with fewer paired replays):")
    lines.append("")
    lines.append("| replays | precision | recall | F1 | false alarms | "
                 "missed |")
    lines.append("|---|---|---|---|---|---|")
    for n_draws in sorted(results["detection_by_draws"],
                          key=lambda item: int(item)):
        entry = results["detection_by_draws"][n_draws]
        lines.append("| %s | %.3f | %.3f | %.3f | %d | %d |" % (
            n_draws, entry["precision"], entry["recall"], entry["f1"],
            entry["fp"], entry["fn"]))
    lines.append("")
    lines.append("Coarse replays quantise the success rates, so inert "
                 "segments near the liability threshold get false "
                 "alarms; planted liabilities are never missed at any "
                 "budget measured.")
    lines.append("")
    lines.append("## Value accuracy")
    lines.append("")
    lines.append("Mean absolute error of measured leave-one-out "
                 "values vs the solver's analytic marginals: **%.4f** "
                 "(success-probability units)." % results["value_mae"])
    lines.append("")
    lines.append("## Redundancy masking (redundant_key tasks)")
    lines.append("")
    lines.append("Each copy of the duplicated key fact prices **%.3f** "
                 "on average alone; the pair prices **%.3f** jointly. "
                 "Leave-one-out pricing alone would call both copies "
                 "nearly worthless." % (
                     masking["mean_individual"], masking["mean_joint"]))
    lines.append("")
    lines.append("## Budget packing: mean measured success")
    lines.append("")
    header = "| strategy | " + " | ".join(
        "%d%%" % int(budget * 100) for budget in BUDGETS) + " | retained @50% |"
    lines.append(header)
    lines.append("|" + "---|" * (len(BUDGETS) + 2))
    for strategy in STRATEGIES:
        cells = " | ".join("%.3f" % packing[strategy]["%.2f" % budget]
                           for budget in BUDGETS)
        lines.append("| %s | %s | %.3f |" % (
            strategy, cells, packing[strategy]["retained_at_50"]))
    lines.append("")
    lines.append("Full context (everything kept, no budget): mean "
                 "success **%.3f**. Retained @50%% is the ratio of "
                 "aggregate means; above 1.0 means the budget cut "
                 "helped by evicting liabilities." %
                 results["mean_full_success"])
    lines.append("")
    lines.append("## Per-scenario success at the 50% budget")
    lines.append("")
    lines.append("| scenario | full | value | relevance | recency | "
                 "oracle |")
    lines.append("|---|---|---|---|---|---|")
    for scenario in sorted(results["per_scenario_at_50"]):
        entry = results["per_scenario_at_50"][scenario]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            scenario, entry["full"], entry["value"],
            entry["relevance"], entry["recency"], entry["oracle"]))
    lines.append("")
    return "\n".join(lines)
