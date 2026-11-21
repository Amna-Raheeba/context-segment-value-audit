"""Streamlit dashboard for context-segment-value-audit.

Three tabs:
1. Audit -- pick a scenario family and seed; the task is generated
   and priced live, with the segment price table, the
   negative-value register, and the duplicate probes.
2. Packing -- one task under a token-budget slider: which segments
   each strategy keeps, and the measured success of each packed
   context against the full context.
3. Evaluation -- the shipped aggregate results from
   results/evaluation.json: packing success per strategy and
   budget, negative-value detection, replay-budget sensitivity,
   and redundancy masking.

Run:  streamlit run app.py
"""

import io
import json
import os

import pandas as pd
import streamlit as st

from context_segment_value.audit import audit_task
from context_segment_value.generate import SCENARIOS, build_task
from context_segment_value.packing import (BUDGETS, STRATEGIES,
                                           pack_and_measure)
from context_segment_value.solver import Solver

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_results():
    path = os.path.join(HERE, "results", "evaluation.json")
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def _audit_cached(scenario, seed):
    task = build_task(scenario, seed)
    audit = audit_task(task)
    rows = []
    for row in audit.rows:
        seg = row.segment
        rows.append({
            "id": seg.segment_id,
            "role": seg.role,
            "planted class": seg.value_class,
            "tokens": seg.tokens,
            "value": round(row.value, 3),
            "per-token": round(row.density, 4),
            "verdict": row.verdict,
        })
    probes = []
    for probe in audit.probes:
        probes.append({
            "pair": "%s + %s" % (probe.id_a, probe.id_b),
            "similarity": round(probe.similarity, 2),
            "individual": "%+.3f / %+.3f" % (probe.value_a, probe.value_b),
            "joint": round(probe.joint_value, 3),
            "masking": probe.masked,
        })
    return {
        "task_id": task.task_id,
        "query": task.query,
        "total_tokens": task.total_tokens,
        "full_success": audit.full_success,
        "rows": rows,
        "probes": probes,
    }


@st.cache_data(show_spinner=False)
def _packing_cached(scenario, seed, budget):
    task = build_task(scenario, seed)
    solver = Solver()
    out = {}
    for strategy in STRATEGIES:
        packed, tokens, success = pack_and_measure(
            task, strategy, budget, solver=solver)
        out[strategy] = {
            "tokens": tokens,
            "success": round(success, 3),
            "kept": ", ".join(packed) if packed else "(nothing)",
        }
    full = solver.success_rate(task, task.segment_ids())
    return out, round(full, 3)


def tab_audit():
    scenario = st.selectbox("Scenario family", list(SCENARIOS), index=2)
    seed = st.slider("Seed", 0, 7, 3)
    data = _audit_cached(scenario, seed)
    st.markdown("**%s** -- %s" % (data["task_id"], data["query"]))
    st.markdown("Full-context measured success: **%.3f** across %d "
                "tokens" % (data["full_success"], data["total_tokens"]))
    frame = pd.DataFrame(data["rows"])
    st.dataframe(frame, use_container_width=True)
    chart = frame.set_index("id")[["value"]]
    st.bar_chart(chart)
    liabilities = [row for row in data["rows"]
                   if row["verdict"] == "liability"]
    st.subheader("Negative-value register")
    if liabilities:
        st.dataframe(pd.DataFrame(liabilities),
                     use_container_width=True)
    else:
        st.write("No liabilities in this task.")
    st.subheader("Duplicate probes")
    if data["probes"]:
        st.dataframe(pd.DataFrame(data["probes"]),
                     use_container_width=True)
    else:
        st.write("No near-duplicate pairs in this task.")


def tab_packing():
    scenario = st.selectbox("Scenario family", list(SCENARIOS), index=5,
                            key="packing_scenario")
    seed = st.slider("Seed", 0, 7, 3, key="packing_seed")
    budget = st.select_slider("Token budget", options=list(BUDGETS),
                              value=0.50,
                              format_func=lambda b: "%d%%" % int(b * 100))
    out, full = _packing_cached(scenario, seed, budget)
    st.markdown("Full-context success: **%.3f**" % full)
    rows = []
    for strategy in STRATEGIES:
        entry = dict(out[strategy])
        entry["strategy"] = strategy
        rows.append(entry)
    frame = pd.DataFrame(rows)[["strategy", "tokens", "success", "kept"]]
    st.dataframe(frame, use_container_width=True)
    st.bar_chart(frame.set_index("strategy")[["success"]])


def tab_evaluation():
    results = _load_results()
    if results is None:
        st.write("No shipped results found. Run: python -m "
                 "context_segment_value evaluate --out results")
        return
    st.markdown("%d seeded tasks, planted ground truth" %
                results["n_tasks"])
    packing = results["packing"]
    rows = []
    for strategy in STRATEGIES:
        row = {"strategy": strategy}
        for budget in BUDGETS:
            row["%d%%" % int(budget * 100)] = round(
                packing[strategy]["%.2f" % budget], 3)
        row["retained @50%"] = round(
            packing[strategy]["retained_at_50"], 3)
        rows.append(row)
    st.subheader("Budget packing: mean measured success")
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True)
    st.bar_chart(frame.set_index("strategy")[["50%"]])
    st.markdown("Full-context mean success: **%.3f**" %
                results["mean_full_success"])
    det = results["detection"]
    st.subheader("Negative-value detection (400 paired replays)")
    st.markdown("Precision **%.3f** -- recall **%.3f** -- F1 **%.3f** "
                "(%d planted negatives)" % (
                    det["precision"], det["recall"], det["f1"],
                    det["tp"] + det["fn"]))
    st.subheader("Detection vs replay budget")
    draw_rows = []
    for n_draws in sorted(results["detection_by_draws"],
                          key=lambda item: int(item)):
        entry = dict(results["detection_by_draws"][n_draws])
        entry["replays"] = int(n_draws)
        draw_rows.append(entry)
    st.dataframe(pd.DataFrame(draw_rows)[
        ["replays", "precision", "recall", "f1", "fp", "fn"]],
        use_container_width=True)
    masking = results["masking"]
    st.subheader("Redundancy masking")
    st.markdown("A duplicated key fact prices **%.3f** per copy but "
                "%.3f jointly (%d tasks)." % (
                    masking["mean_individual"], masking["mean_joint"],
                    masking["tasks"]))
    st.markdown("Measured values vs analytic marginals: MAE **%.4f**"
                % results["value_mae"])


def main():
    st.set_page_config(page_title="Context Segment Value Audit")
    st.title("Context Segment Value Audit")
    st.write("Every segment of an agent's context, priced by its "
             "causal leave-one-out effect on task success -- "
             "including the segments whose value is negative.")
    audit_tab, packing_tab, evaluation_tab = st.tabs(
        ["Audit", "Packing", "Evaluation"])
    with audit_tab:
        tab_audit()
    with packing_tab:
        tab_packing()
    with evaluation_tab:
        tab_evaluation()


if __name__ == "__main__":
    main()
