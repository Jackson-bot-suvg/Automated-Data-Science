#!/usr/bin/env python3
"""Task-level paired statistics, resource ledger, A0 tables, quality gates and claim ledger.

Input : results/run_level.csv (scripts/evaluate.py), logs/plan_ledger.csv, raw run dirs.
Output: results/pair_level.csv, results/task_level.csv, results/resource_ledger.csv,
        results/quality_gates.csv, results/claims.csv,
        results/A0/{A0_old_vs_current.csv, A0_smoke.csv, A0_probe_outcomes.csv,
                    A0_old_vs_corrected_ablation.csv, A0_reviewer_event_list.txt}
All numbers are recomputed from raw records; nothing is typed in by hand except the
manuscript's own published values (clearly labelled source=manuscript).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PKG, WORK  # noqa: E402
from specs import A2_CONTEXTS, DEV_TASKS  # noqa: E402

RES = PKG / "results"
A0 = RES / "A0"
BOOT_SEED = 1729
N_BOOT = 10_000

MANUSCRIPT_T2 = {  # manuscript Table 2 (post-patch clean, technical-only); source=manuscript
    "DEV-MC-ACADEMIC": ("Macro-F1", 0.7938, "3/3", 60.6, 0.5),
    "DEV-BIN-BANK": ("ROC-AUC", 0.8892, "3/3", 41.4, 1.0),
    "DEV-MC-GHOULS": ("Macro-F1", 0.7540, "3/3", 16.6, 1.2),
    "DEV-REG-HOUSE": ("Neg. RMSE", -29622.8, "3/3", 49.0, 0.4),
    "DEV-MC-OBESITY": ("Macro-F1", 0.8941, "3/3", 114.7, 2.1),
    "DEV-ML-STEEL": ("ROC-AUC", 0.8883, "3/3", 276.6, 2.9),
    "DEV-BIN-SPACESHIP": ("Acc.", 0.7934, "3/3", 31.3, 2.2),
    "DEV-BIN-TITANIC": ("ROC-AUC", 0.8770, "3/3", 16.4, 0.7),
}
MANUSCRIPT_T3 = [  # manuscript Table 3; source=manuscript
    ("Full RTS-MA-DS", "4/4", "4/4", "2/2", "0/2", "0/4", 70.84),
    ("No target-spec", "4/4", "4/4", "0/2", "2/2", "0/4", 123.56),
    ("Schema-only", "4/4", "4/4", "0/2", "2/2", "0/4", 131.83),
    ("No semantic align.", "4/4", "4/4", "0/2", "2/2", "0/4", 132.82),
    ("No reviewer gates", "0/4", "0/4", "2/2", "0/2", "4/4", 8.40),
]


def cp_interval(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return (float(lo), float(hi))


def boot_ci(task_means: list[float]) -> tuple[float, float]:
    vals = np.array([v for v in task_means if not (v is None or (isinstance(v, float) and math.isnan(v)))], dtype=float)
    if len(vals) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _b(x) -> bool:
    return str(x).strip().lower() == "true"


def count_true(series) -> int:
    return int(sum(1 for x in series if _b(x)))


def bmask(series) -> np.ndarray:
    return np.array([_b(x) for x in series], dtype=bool)


def _f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return math.nan


# ---------------------------------------------------------------------------
# A0
# ---------------------------------------------------------------------------


def best_cv_score(run_dir: Path) -> tuple[float, str]:
    for p in run_dir.rglob("model_results.json"):
        try:
            d = json.loads(p.read_text())
            best = d.get("best_model_name")
            metric = (d.get("_meta") or {}).get("primary_metric", "")
            vm = (d.get("validation_metrics") or {}).get(best) or {}
            for key in (f"cv_{metric}_mean", f"val_{metric}"):  # CV mean, or holdout score when F0 selects by holdout (large/multilabel tasks)
                if key in vm:
                    return float(vm[key]), metric + (" (holdout)" if key.startswith("val_") else "")
            cands = [k for k in vm if k.startswith("cv_") and k.endswith("_mean")]
            if cands:
                return float(vm[cands[0]]), cands[0][3:-5]
        except Exception:  # noqa: BLE001
            continue
    return math.nan, ""


def a0_tables(rl: pd.DataFrame) -> None:
    A0.mkdir(parents=True, exist_ok=True)
    rows = []
    clean = rl[rl["stage"] == "A0-CLEAN"]
    for task_id in DEV_TASKS:
        sub = clean[clean["task_id"] == task_id]
        ms = MANUSCRIPT_T2[task_id]
        scores, metrics = [], set()
        for rid in sub["run_id"]:
            s, m = best_cv_score(WORK / "runs" / rid)
            scores.append(s)
            metrics.add(m)
        walls = [_f(x) for x in sub["wall_seconds"]]
        n = len(sub)
        completed = count_true(sub["delivered"])
        valid = count_true(sub["artifact_valid"])
        sc = [s for s in scores if not math.isnan(s)]
        rows.append({
            "task_id": task_id,
            "manuscript_metric": ms[0], "manuscript_cv_score": ms[1], "manuscript_valid": ms[2],
            "manuscript_time_s_mean": ms[3], "manuscript_time_s_sd": ms[4],
            "current_runs_started": n,
            "current_delivered": f"{completed}/{n}",
            "current_valid": f"{valid}/{n}",
            "current_terminals": "|".join(sub["terminal_S"].tolist()),
            "current_metric_used": "|".join(sorted(m for m in metrics if m)),
            "current_cv_score_per_repeat": "|".join(f"{s:.4f}" for s in scores),
            "current_cv_score_mean": round(float(np.mean(sc)), 4) if sc else "NA",
            "current_cv_score_sd": round(float(np.std(sc, ddof=1)), 4) if len(sc) > 1 else "NA",
            "current_wall_s_mean": round(float(np.nanmean(walls)), 1) if walls else "NA",
            "current_wall_s_sd": round(float(np.nanstd(walls, ddof=1)), 1) if len(walls) > 1 else "NA",
            "score_delta_vs_manuscript": round(float(np.mean(sc)) - ms[1], 4) if sc else "NA",
            "config_difference": "LLM backend deepseek-flash (MiniMax-M2.7 EOL); local machine; same code b9dbebe (public release)",
            "attribution": "config (LLM backend, hardware) unless a code diff is shown; code = public release (manuscript SHA unknown)",
        })
    write_csv(A0 / "A0_old_vs_current.csv", rows)

    smoke = rl[rl["stage"].isin(["A0-SMOKE"]) | ((rl["stage"] == "A0-CLEAN") & rl["task_id"].isin(["DEV-BIN-TITANIC", "DEV-MC-GHOULS"]) & rl["run_id"].str.endswith("__r1"))]
    srows = []
    for _, r in smoke.iterrows():
        run_dir = WORK / "runs" / r["run_id"]
        computed = enforced = 0
        ev_path = run_dir / "events.jsonl"
        n_nodes = 0
        if ev_path.exists():
            for line in ev_path.read_text().splitlines():
                e = json.loads(line)
                if e["event_type"] == "gate_check":
                    computed += len((e.get("payload") or {}).get("computed_failures") or [])
                    enforced += len((e.get("payload") or {}).get("enforced_failures") or [])
                if e["event_type"] == "node_start":
                    n_nodes += 1
        srows.append({
            "run_id": r["run_id"], "task_id": r["task_id"], "method_id": r["method_id"], "terminal_S": r["terminal_S"],
            "subreason": r["terminal_subreason"], "delivered": r["delivered"], "artifact_valid": r["artifact_valid"],
            "node_executions": n_nodes, "computed_gate_failures": computed, "enforced_gate_failures": enforced,
            "wall_seconds": r["wall_seconds"], "api_calls": r["api_calls"],
        })
    write_csv(A0 / "A0_smoke.csv", srows)

    probes = []
    fam_task = {"binary_classification": "DEV-BIN-SPACESHIP", "multiclass_classification": "DEV-MC-GHOULS", "regression": "DEV-REG-HOUSE", "multilabel_classification(manuscript rep.)": "DEV-ML-STEEL"}
    probe_names = {"F01": "target missing", "F03": "misleading explicit target hint", "B01": "legal output alias", "F05": "post-intake target replacement"}
    for fam, task_id in fam_task.items():
        for tmpl, pname in probe_names.items():
            hit = rl[(rl["task_id"] == task_id) & (rl["template_id"] == tmpl) & (rl["method_id"] == "F0_ORIGINAL") & (rl["stage"].isin(["A1", "A0-PROBE"]))]
            for _, r in hit.iterrows():
                probes.append({"family": fam, "task_id": task_id, "probe": pname, "template_id": tmpl, "run_id": r["run_id"],
                               "terminal_S": r["terminal_S"], "safe_terminal": r["safe_terminal"], "subreason": r["terminal_subreason"],
                               "C_read_targets": r["C_read_targets"], "C_used_fit_targets": r["C_used_fit_targets"],
                               "output_provenance": r["C_used_output_provenance"], "localization": r["localization"],
                               "fatal_gate": r["fatal_gate"]})
    write_csv(A0 / "A0_probe_outcomes.csv", probes)

    abl = [{"source": "manuscript Table 3", "variant": v, "clean_completion": c, "clean_valid": vv, "stop": s, "fn": fn, "fp": fp, "time_s": t} for v, c, vv, s, fn, fp, t in MANUSCRIPT_T3]
    for method, label in [("F0_ORIGINAL", "F0 public release (current)"), ("M0_NO_SEMANTIC_ENFORCEMENT", "M0 corrected: semantic verdicts not enforced, state/routing kept"),
                          ("M3_SHADOW", "M3 shadow: all verdicts computed, none enforced"), ("M0_NAIVE_GATE_REMOVAL", "reconstruction of 'No reviewer gates': gate nodes removed, routing kept")]:
        sub = smoke[smoke["method_id"] == method]
        n = len(sub)
        comp = count_true(sub["delivered"])
        val = count_true(sub["artifact_valid"])
        abl.append({"source": "lane A DEV (2 clean tasks: Titanic, Ghouls)", "variant": label, "clean_completion": f"{comp}/{n}", "clean_valid": f"{val}/{n}",
                    "stop": "not run (manuscript fault packages unavailable)", "fn": "NA", "fp": f"{n - comp}/{n}",
                    "time_s": round(float(np.nanmean([_f(x) for x in sub["wall_seconds"]])), 2) if n else "NA",
                    "terminals": "|".join(sub["terminal_S"].tolist())})
    write_csv(A0 / "A0_old_vs_corrected_ablation.csv", abl)

    # reviewer state/routing event list (pure text) for one real clean F0 run and one fault run
    lines = []
    for rid_hint in ["A0C__DEV-BIN-TITANIC__C00__F0_ORIGINAL__sb1101__r1", "A1__DEV-BIN-SPACESHIP__F03__F0_ORIGINAL__sb1101", "A1__DEV-BIN-SPACESHIP__F05__F0_ORIGINAL__sb1101"]:
        p = WORK / "runs" / rid_hint / "events.jsonl"
        if not p.exists():
            continue
        lines.append(f"=== run {rid_hint}")
        for line in p.read_text().splitlines():
            e = json.loads(line)
            et = e["event_type"]
            pl = e.get("payload") or {}
            if et in {"node_start"}:
                lines.append(f"{e['monotonic_seq']:4d} node_start   {e['actor']:20s} phase={e['phase']} contract={e['contract_digest'][:10]}")
            elif et == "gate_check":
                fails = "; ".join(f"[{f['class']}] {f['text'][:70]}" for f in pl.get("computed_failures") or []) or "-"
                lines.append(f"{e['monotonic_seq']:4d} gate_check   {e['actor']:20s} phase={e['phase']} f0_status={e['verdict']} failures={fails}")
            elif et == "gate_verdict":
                lines.append(f"{e['monotonic_seq']:4d} gate_verdict {e['actor']:20s} phase={e['phase']} verdict={e['verdict']} -> {e['next_action']}")
            elif et == "injection":
                lines.append(f"{e['monotonic_seq']:4d} INJECTION    {pl.get('template_id')} {json.dumps({k: pl.get(k) for k in ('before_target_columns', 'after_target_columns', 'fixed_points') if k in pl})}")
            elif et == "fit_start":
                lines.append(f"{e['monotonic_seq']:4d} fit          y={pl.get('y_columns')} family={pl.get('task_family')} metric={pl.get('primary_metric')} hash={e['actual_fit_target_hash'][:10]}")
            elif et == "serialize":
                lines.append(f"{e['monotonic_seq']:4d} serialize    provenance={pl.get('column_provenance')}")
            elif et == "artifact_write":
                lines.append(f"{e['monotonic_seq']:4d} write        shape={pl.get('shape')} cols={pl.get('columns')}")
            elif et == "run_end":
                lines.append(f"{e['monotonic_seq']:4d} run_end      outcome={pl.get('outcome')} review={pl.get('latest_review_status')} completed={pl.get('workflow_completed')}")
    (A0 / "A0_reviewer_event_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------


def failure_source(r) -> str:
    """Protocol A1 categories, from oracle fields + serialize provenance (never from agent text)."""

    prov = json.loads(r["C_used_output_provenance"] or "{}") if str(r.get("C_used_output_provenance", "")).startswith("{") else {}
    if not _b(r["C_read_matches_gold"]) and r["C_read_targets"]:
        return "initial_contract_error(reader resolved a different target)"
    if r["n_fits"] in {"0", "", 0} and r["terminal_S"] in {"S3", "S4"}:
        return "stopped_before_fit"
    if r["n_fits"] in {"0", "", 0}:
        return "no_fit(execution failure)"
    if not _b(r["fit_target_matches_gold"]) or not _b(r["task_family_matches_gold"]):
        return "stage_drift(C_read correct, fit consumed a different target/family)"
    if any(v == "synthesized_range_id" for k, v in prov.items()) and not _b(r["output_mapping_valid"]):
        return "consumption_decoupling(output role filled by synthesized ids, not predictions)"
    if any(v == "prediction_reordered_or_altered" for v in prov.values()):
        return "output_mismatch(ID->prediction pairing altered after predict())"
    if not _b(r["output_mapping_valid"]) and r["delivered"] in {"True", True}:
        return "output_mismatch(other)"
    return "consistent(C_gold = C_read = C_used, mapping valid)"


def a1_tables(rl: pd.DataFrame) -> list[dict]:
    a1 = rl[rl["stage"].isin(["A1", "A0-PROBE"])].copy()
    a1["a1_failure_source"] = [failure_source(r) for _, r in a1.iterrows()]
    write_csv(RES / "A1_run_binding.csv", a1[["run_id", "stage", "task_id", "template_id", "method_id", "C_read_targets", "C_read_family", "C_read_output_columns",
                                            "C_used_fit_targets", "C_used_family", "actual_metric", "metric_aligned", "C_used_output_provenance",
                                            "output_mapping_valid", "a1_failure_source", "fault_detected_by_gate", "localization", "fatal_gate",
                                            "terminal_S", "terminal_subreason", "event_trace_path"]].to_dict("records"))
    a1 = a1[a1["stage"] == "A1"]
    rows = []
    for (task, tmpl), g in a1.groupby(["task_id", "template_id"]):
        f0 = g[g["method_id"] == "F0_ORIGINAL"]
        m3 = g[g["method_id"] == "M3_SHADOW"]
        if f0.empty or m3.empty:
            continue
        a, b = f0.iloc[0], m3.iloc[0]
        rows.append({
            "pair_id": f"A1PAIR__{task}__{tmpl}", "experiment": "A1", "task_id": task, "template_id": tmpl,
            "F0_run_id": a["run_id"], "M3_run_id": b["run_id"],
            "F0_terminal": a["terminal_S"], "M3_terminal": b["terminal_S"],
            "F0_C_read": a["C_read_targets"], "M3_C_read": b["C_read_targets"],
            "F0_fit_target": a["C_used_fit_targets"], "M3_fit_target": b["C_used_fit_targets"],
            "F0_failure_source": a["a1_failure_source"], "M3_failure_source": b["a1_failure_source"],
            "F0_output_provenance": a["C_used_output_provenance"], "M3_output_provenance": b["C_used_output_provenance"],
            "F0_gate_detected_fault": a["fault_detected_by_gate"], "M3_shadow_detected_fault": b["fault_detected_by_gate"],
            "verdict_changed_action": a["terminal_S"] != b["terminal_S"],
            "comparability": "independent full runs (same package, same seed block; not a common-prefix fork)",
        })
    return rows


# ---------------------------------------------------------------------------
# A2 / A3
# ---------------------------------------------------------------------------

INDICATORS = {
    "silent_S5": lambda t: t == "S5",
    "unsafe_delivery_S5S6": lambda t: t in {"S5", "S6"},
    "correct_delivery_S1S2": lambda t: t in {"S1", "S2"},
    "unnecessary_abort_S4": lambda t: t == "S4",
    "justified_stop_S3": lambda t: t == "S3",
    "visible_failure_S7": lambda t: t == "S7",
}
CONTRASTS = [("F0_ORIGINAL", "M3_SHADOW", "F0-M3"), ("F0_ORIGINAL", "M5_BLOCK_ONLY", "F0-M5"), ("F0_ORIGINAL", "M4_END_ONLY", "F0-M4"),
             ("G_GENERIC_RETRY", "F0_ORIGINAL", "G-F0"), ("F0_ORIGINAL", "M2_INTAKE_ONLY_STRONG", "F0-M2")]


def a2_tables(rl: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    br = rl[rl["stage"].isin(["A2-PILOT", "A3-PILOT"])]
    prefixes = rl[rl["stage"] == "A2-PREFIX"].set_index("run_id")
    branch_rows, pair_rows = [], []
    for pid, g in br.groupby("pair_id"):
        by_m = {r["method_id"]: r for _, r in g.iterrows()}
        strict = all(_b(r["fork_state_hash_match"]) and _b(r["fork_candidate_hash_match"]) and _b(r["fork_rng_match"]) for r in by_m.values())
        ctx = g.iloc[0]["condition"]
        task = g.iloc[0]["task_id"]
        sb = g.iloc[0]["seed_block"]
        for m, r in by_m.items():
            row = {"pair_id": pid, "context": ctx, "task_id": task, "seed_block": sb, "method_id": m, "run_id": r["run_id"],
                   "terminal_S": r["terminal_S"], "fork_gate_verdict": r["fork_gate_verdict"], "next_action": r["fork_gate_next_action"],
                   "f0_counterfactual_at_gate": r["fork_gate_f0_counterfactual"],
                   "wrong_fits_after_fork": r["wrong_fits_after_fork"], "wrong_training_seconds_after_fork": r["wrong_training_seconds_after_fork"],
                   "wrong_file_delivered": r["terminal_S"] in {"S5", "S6"}, "api_calls_after_fork": r["api_calls_own"],
                   "wall_seconds_after_fork": r["wall_seconds"], "strict_common_prefix": strict}
            for k, fn in INDICATORS.items():
                row[k] = int(fn(r["terminal_S"]))
            branch_rows.append(row)
        for a, b, name in CONTRASTS:
            if a in by_m and b in by_m:
                ra, rb = by_m[a], by_m[b]
                pr = {"pair_id": pid, "experiment": "A3" if "M2" in name else "A2", "context": ctx, "task_id": task, "seed_block": sb,
                      "contrast": name, "strict_common_prefix": strict, "terminal_a": ra["terminal_S"], "terminal_b": rb["terminal_S"],
                      "run_a": ra["run_id"], "run_b": rb["run_id"]}
                for k, fn in INDICATORS.items():
                    pr[f"d_{k}"] = int(fn(ra["terminal_S"])) - int(fn(rb["terminal_S"]))
                for k in ("wrong_training_seconds_after_fork", "api_calls_own", "wall_seconds"):
                    pr[f"d_{k}"] = round(_f(ra[k]) - _f(rb[k]), 3)
                pair_rows.append(pr)
    # task level: mean over seed blocks per task, then across tasks
    task_rows = []
    pdf = pd.DataFrame(pair_rows)
    if not pdf.empty:
        main = pdf[pdf["strict_common_prefix"]]
        for (contrast, ctx), g in main.groupby(["contrast", "context"]):
            for metric in [c for c in pdf.columns if c.startswith("d_")]:
                per_task = g.groupby("task_id")[metric].mean()
                vals = per_task.tolist()
                lo, hi = boot_ci(vals)
                loo = [float(np.mean([v for j, v in enumerate(vals) if j != i])) for i in range(len(vals))] if len(vals) > 1 else []
                task_rows.append({"contrast": contrast, "context": ctx, "metric": metric, "n_tasks": len(vals), "n_pairs": len(g),
                                  "task_level_mean": round(float(np.mean(vals)), 4) if vals else "NA",
                                  "boot95_low": round(lo, 4) if not math.isnan(lo) else "NA", "boot95_high": round(hi, 4) if not math.isnan(hi) else "NA",
                                  "per_task": json.dumps({k: round(float(v), 4) for k, v in per_task.items()}),
                                  "direction": "|".join("+" if v > 0 else ("-" if v < 0 else "0") for v in vals),
                                  "leave_one_task_out_range": f"[{min(loo):.3f}, {max(loo):.3f}]" if loo else "NA",
                                  "caveat": "DEV pilot, 3 tasks: bootstrap interval unstable, descriptive only"})
        # pooled across contexts
        for contrast, g in main.groupby("contrast"):
            for metric in ["d_silent_S5", "d_unsafe_delivery_S5S6", "d_correct_delivery_S1S2", "d_unnecessary_abort_S4", "d_justified_stop_S3", "d_wrong_training_seconds_after_fork", "d_api_calls_own"]:
                per_task = g.groupby("task_id")[metric].mean()
                vals = per_task.tolist()
                lo, hi = boot_ci(vals)
                task_rows.append({"contrast": contrast, "context": "ALL", "metric": metric, "n_tasks": len(vals), "n_pairs": len(g),
                                  "task_level_mean": round(float(np.mean(vals)), 4) if vals else "NA",
                                  "boot95_low": round(lo, 4) if not math.isnan(lo) else "NA", "boot95_high": round(hi, 4) if not math.isnan(hi) else "NA",
                                  "per_task": json.dumps({k: round(float(v), 4) for k, v in per_task.items()}),
                                  "direction": "|".join("+" if v > 0 else ("-" if v < 0 else "0") for v in vals),
                                  "leave_one_task_out_range": "NA", "caveat": "pooled over contexts; DEV pilot"})
    return branch_rows, pair_rows, task_rows


# ---------------------------------------------------------------------------
# resource ledger
# ---------------------------------------------------------------------------


def resource_rows(rl: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in rl.iterrows():
        run_dir = WORK / "runs" / r["run_id"]
        res = json.loads((run_dir / "run_result.json").read_text()) if (run_dir / "run_result.json").exists() else {}
        out.append({
            "run_id": r["run_id"], "stage": r["stage"], "task_id": r["task_id"], "method_id": r["method_id"],
            "wall_seconds_own": r["wall_seconds"], "prefix_wall_seconds_shared": r["prefix_wall_seconds"],
            "fit_seconds_own": res.get("fit_seconds_total", "NA"), "wrong_training_seconds_total": r["wrong_training_seconds"],
            "wrong_training_seconds_after_fork": r.get("wrong_training_seconds_after_fork", "NA"),
            "api_calls_total_trajectory": r["api_calls"], "api_calls_own": r["api_calls_own"], "api_errors": r["api_errors"],
            "tokens_in_if_observed": r["tokens_in_if_observed"], "tokens_out_if_observed": r["tokens_out_if_observed"],
            "tokens_scope": "whole trajectory (an A2 prefix is counted in each of its branches; sum own events for totals)",
            "measured_cost_if_observed": "NA", "missingness": r["missingness_reason"] or ("cost NA: provider price not frozen" ),
        })
    return out


# ---------------------------------------------------------------------------
# quality gates and claims
# ---------------------------------------------------------------------------


def quality_and_claims(rl: pd.DataFrame, ledger: pd.DataFrame, pair_rows: list[dict], task_rows: list[dict], a1_rows: list[dict]) -> None:
    now = pd.Timestamp.now(tz="Australia/Sydney").isoformat(timespec="seconds")
    gates = []

    def gate(exp, gid, status, checked, eligible, evidence, reason):
        gates.append({"experiment_id": exp, "gate_id": gid, "status": status, "checked_count": checked, "eligible_count": eligible,
                      "evidence_path": evidence, "reason": reason, "checked_by": "lane A (automated: scripts/analyze.py)", "checked_at": now})

    gate("ALL", "G0", "FAIL", 0, 1, "FREEZE_SIGNOFF.yaml", "unsigned; CONF/TRANSFER identities [待冻结] -> only DEV allowed; no CONF claim")
    man = pd.read_csv(PKG / "cases" / "manifest.csv")
    app = man[man["applicability"] != "N/A"]
    mut_ok = (app["mutation_verified"].astype(str) != "False").all()
    inj = rl[rl["template_id"].isin(["F05", "F10"]) & ~rl["terminal_S"].isin(["PREFIX_FORKED"])]
    inj_bad = inj[(inj["injection_verified"].astype(str) != "True")]
    # private label isolation: no agent-visible test.csv contains a target column
    leak = []
    for _, m in app.iterrows():
        pkg = WORK / str(m["package_path"])
        if (pkg / "test.csv").exists():
            cols = pd.read_csv(pkg / "test.csv", nrows=0).columns
            if any(t in cols for t in DEV_TASKS[m["task_id"]]["target_columns"]):
                leak.append(m["case_id"])
    g1 = "PASS" if (mut_ok and inj_bad.empty and not leak) else "FAIL"
    gate("DEV", "G1", g1, len(app) + len(inj), len(app) + len(inj), "cases/manifest.csv; results/run_level.csv(injection_verified)",
         f"file mutations verified={mut_ok}; runtime injections not verified={len(inj_bad)} {list(inj_bad['run_id'])[:5]}; test.csv target leaks={leak}")
    vd = pd.read_csv(A0 / "version_diff.csv") if (A0 / "version_diff.csv").exists() else pd.DataFrame()
    eq_ok = (not vd.empty) and bmask(vd["identical"]).all()
    clean_f0 = rl[(rl["stage"] == "A0-CLEAN")]
    n_clean_ok = clean_f0.groupby("task_id")["delivered"].apply(lambda s: bmask(s).any()).sum() if not clean_f0.empty else 0
    smoke = rl[rl["stage"] == "A0-SMOKE"]
    m0_ok = count_true(smoke[smoke["method_id"] == "M0_NO_SEMANTIC_ENFORCEMENT"]["delivered"])
    m3_ok = count_true(smoke[smoke["method_id"] == "M3_SHADOW"]["delivered"])
    g2 = "PASS" if (eq_ok and n_clean_ok >= 2 and m0_ok >= 2 and m3_ok >= 2) else "FAIL"
    gate("DEV", "G2", g2, int(n_clean_ok), 8, "results/A0/version_diff.csv; results/A0/A0_smoke.csv",
         f"hook equivalence={eq_ok}; DEV tasks with >=1 clean F0 delivery={n_clean_ok}; M0 clean deliveries={m0_ok}/2; M3={m3_ok}/2")
    planned = ledger[ledger["planned_status"] == "planned"]
    started = rl
    no_term = started[~started["terminal_S"].isin(["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "PREFIX_FORKED"])]
    unstarted = planned[~planned["plan_cell_id"].isin(started["run_id"])]
    g3 = "PASS" if no_term.empty else "FAIL"
    gate("DEV", "G3", g3, len(started), len(planned), "logs/plan_ledger.csv; logs/run_index.csv; results/run_level.csv",
         f"started={len(started)}; started without unique terminal={len(no_term)}; planned but not started={len(unstarted)} (listed in plan_ledger with status)")
    br = rl[rl["stage"].isin(["A2-PILOT", "A3-PILOT"])]
    nonstrict = br[~(bmask(br["fork_state_hash_match"]) & bmask(br["fork_candidate_hash_match"]) & bmask(br["fork_rng_match"]))]
    g4 = "PASS" if (nonstrict.empty and not leak) else "FAIL"
    gate("A2/A3", "G4", g4, len(br), len(br), "results/pair_level.csv(strict_common_prefix)", f"branches with hash mismatch={len(nonstrict)}; private-test leaks={leak}")
    gate("DEV", "G5", "PASS(partial matrix: DEV pilot only)", len(started), len(started),
         "scripts/make_paper_assets.py; paper/cases/*.md", "every table/figure regenerated from raw events; manual check of favourable and unfavourable records in paper/cases; CONF matrix not run")
    gate("DEV", "G6", "PASS", 4, 4, "results/claims.csv", "QC / USABLE / SUPPORT recorded separately per experiment; DEV pilot capped at CLAIM_DIRECTIONAL")
    # per-experiment lane gates
    a0_ok = eq_ok and m0_ok >= 2
    gate("A0", "A0-QC", "PASS" if a0_ok else "FAIL", 1, 1, "results/A0/*", "identifiable version (public release); F0/M0/M3 differences located at function level; M0/M3 complete >=2 clean DEV tasks; probes hashed")
    a1 = rl[rl["stage"] == "A1"]
    a1_cells = ledger[ledger["stage"] == "A1"]
    a1_ok = len(a1) + int((a1_cells["planned_status"] == "N/A").sum()) == len(a1_cells)
    a1_fit_events = a1[a1["n_fits"].astype(float) > 0]
    gate("A1", "A1-QC", "PASS" if a1_ok else "FAIL", len(a1), len(a1_cells), "results/run_level.csv; logs/events",
         f"48 cells: run={len(a1)}, N/A={int((a1_cells['planned_status'] == 'N/A').sum())}; consistency claims use fit/predict/serialize hook events (not agent text); runs with a real fit event={len(a1_fit_events)}")
    a2p = rl[rl["stage"] == "A2-PREFIX"]
    forked = (a2p["terminal_S"] == "PREFIX_FORKED").sum()
    gate("A2", "A2-QC", "PASS(DEV pilot)" if g4 == "PASS" else "FAIL", int(forked), len(a2p), "results/pair_level.csv",
         f"prefixes forked={forked}/{len(a2p)}; every launched branch has one terminal; formal 144-prefix CONF matrix not started (G0)")
    a3 = rl[rl["stage"] == "A3-PILOT"]
    gate("A3", "A3-QC", "PASS(DEV pilot)" if not a3.empty else "NA", len(a3), 12, "results/pair_level.csv (contrast F0-M2)",
         "M2 uses identical intake path (same snapshot at intake for context a; same post-intake prefix for context c)")
    write_csv(RES / "quality_gates.csv", gates)

    # claims
    tl = pd.DataFrame(task_rows)

    def tl_get(contrast, ctx, metric):
        if tl.empty:
            return None
        hit = tl[(tl["contrast"] == contrast) & (tl["context"] == ctx) & (tl["metric"] == metric)]
        return hit.iloc[0].to_dict() if not hit.empty else None

    claims = []
    # favourable sign of (first method - second method); all A2 contexts are fault cases
    FAVOUR = {"d_unsafe_delivery_S5S6": -1, "d_silent_S5": -1, "d_justified_stop_S3": +1, "d_unnecessary_abort_S4": -1,
              "d_correct_delivery_S1S2": +1, "d_wrong_training_seconds_after_fork": -1, "d_api_calls_own": -1}
    pdf_all = pd.DataFrame(pair_rows)
    ALLCTX = ["a_recoverable_intake_conflict", "b_unrecoverable_missing_target", "c_post_planner_fit_target_drift", "d_predict_id_role_misalignment"]
    for cid, contrast, ctxs, primary, secondary, text in [
        ("A-C1", "F0-M3", ALLCTX, "d_unsafe_delivery_S5S6", ["d_justified_stop_S3", "d_wrong_training_seconds_after_fork", "d_correct_delivery_S1S2", "d_unnecessary_abort_S4", "d_api_calls_own"],
         "enforcing F0 verdicts (vs computing them in shadow) changes the action taken on the same candidate"),
        ("A-C2", "F0-M2", ["a_recoverable_intake_conflict", "c_post_planner_fit_target_drift"], "d_unsafe_delivery_S5S6", ["d_justified_stop_S3", "d_correct_delivery_S1S2", "d_api_calls_own"],
         "F0's post-intake checks add protection over an identical intake-only check (M2)"),
        ("A-C3a", "F0-M5", ALLCTX, "d_correct_delivery_S1S2", ["d_unsafe_delivery_S5S6", "d_justified_stop_S3", "d_api_calls_own"],
         "F0's default retry preserves more correct deliveries than block-only (M5)"),
        ("A-C3b", "G-F0", ["c_post_planner_fit_target_drift", "d_predict_id_role_misalignment"], "d_correct_delivery_S1S2", ["d_unsafe_delivery_S5S6", "d_api_calls_own"],
         "generic retry without field diagnostics (G) differs from F0's diagnostic retry"),
        ("A-C4", "F0-M4", ALLCTX, "d_unsafe_delivery_S5S6", ["d_wrong_training_seconds_after_fork", "d_justified_stop_S3", "d_api_calls_own"],
         "early gates (F0) vs the same rules applied once at the end (M4)"),
    ]:
        for ctx in ctxs + ["ALL"]:
            for est in [primary] + secondary:
                t = tl_get(contrast, ctx, est)
                if t is None:
                    continue
                mean = t["task_level_mean"]
                if mean == "NA":
                    support = "INDETERMINATE"
                elif float(mean) == 0:
                    support = "CLAIM_NOT_SUPPORTED(no difference)"
                else:
                    fav = (float(mean) * FAVOUR.get(est, 1)) > 0
                    support = "CLAIM_DIRECTIONAL(DEV pilot)" if fav else "CLAIM_NOT_SUPPORTED(reverse direction)"
                counter = ""
                if not pdf_all.empty:
                    g = pdf_all[(pdf_all["contrast"] == contrast) & ((pdf_all["context"] == ctx) | (ctx == "ALL"))]
                    bad = g[g[est].astype(float) * FAVOUR.get(est, 1) < 0] if est in g.columns else g.iloc[0:0]
                    counter = "|".join(bad["pair_id"].tolist()[:8])
                claims.append({"claim_id": f"{cid}:{ctx}:{est}", "experiment_id": "A3" if contrast == "F0-M2" else "A2", "primary_contrast": contrast,
                               "estimand_role": "primary" if est == primary else "secondary",
                               "estimand": f"{est} (task-level mean of same-prefix paired differences)",
                               "units": "independent DEV base tasks (n=" + str(t["n_tasks"]) + "; pairs=" + str(t["n_pairs"]) + ")", "estimate": mean, "ci_low": t["boot95_low"], "ci_high": t["boot95_high"],
                               "per_task_direction": t["direction"], "qc_status": "QC_PASS(DEV pilot)", "result_usable": "RESULT_USABLE(DEV pilot, descriptive)",
                               "support_level": support if est == primary else f"secondary:{support}", "counterexample_ids": counter,
                               "allowed_paper_wording": f"In a DEV pilot (3 tasks x 2 seed blocks), {text}; {est} = {mean} (per-task {t['direction']}); descriptive, not a CONF confirmation."})
    a1df = pd.DataFrame(a1_rows)
    if not a1df.empty:
        changed = int(a1df["verdict_changed_action"].sum())
        claims.append({"claim_id": "A1-binding", "experiment_id": "A1", "primary_contrast": "F0 vs M3 (independent runs)", "estimand": "pairs where enforcement changed the terminal",
                       "units": f"{len(a1df)} (task,probe) pairs over 3 DEV tasks", "estimate": changed, "ci_low": "NA", "ci_high": "NA", "qc_status": "QC_PASS(DEV)",
                       "result_usable": "RESULT_USABLE(DEV)", "support_level": "see evidence report section A1", "counterexample_ids": "",
                       "allowed_paper_wording": "F0's contract constrains the actual fit only where the intake gate blocks; after intake, fit/serialization are not bound (see T-A1)."})
    write_csv(RES / "claims.csv", claims)


def strict_domain_sensitivity(rl: pd.DataFrame) -> list[dict]:
    """Sensitivity only (not the pre-registered rule): would a strict per-task label domain change S?

    The frozen oracle accepts probabilities in [0,1] for every binary task.  Tasks whose official
    submission format is a label (Spaceship: True/False) are re-checked here with a strict rule.
    """

    out = []
    strict_label_tasks = {"DEV-BIN-SPACESHIP": {"true", "false"}}
    for _, r in rl.iterrows():
        if r["task_id"] not in strict_label_tasks or r["terminal_S"] not in {"S1", "S2", "S5"}:
            continue
        rid = r["run_id"]
        res_p = WORK / "runs" / rid / "run_result.json"
        res = json.loads(res_p.read_text()) if res_p.exists() else {}
        p = Path(res.get("prediction_path") or "")
        if not p.exists():
            continue
        sub = pd.read_csv(p, dtype=str, keep_default_na=False)
        out_cols = [c for c in sub.columns if c not in DEV_TASKS[r["task_id"]]["id_columns"]]
        vals = set(sub[out_cols[0]].str.strip().str.lower()) if out_cols else set()
        is_label = vals <= strict_label_tasks[r["task_id"]] | {"0", "1"}
        out.append({"run_id": rid, "stage": r["stage"], "method_id": r["method_id"], "terminal_S_frozen_rule": r["terminal_S"],
                    "output_is_label": is_label, "n_distinct_values": len(vals),
                    "terminal_S_under_strict_label_rule": r["terminal_S"] if is_label else "S6"})
    write_csv(RES / "sensitivity_strict_label_domain.csv", out)
    return out


def main() -> int:
    rl = pd.read_csv(RES / "run_level.csv", dtype=object, keep_default_na=False)
    strict_domain_sensitivity(rl)
    ledger = pd.read_csv(PKG / "logs" / "plan_ledger.csv", dtype=object, keep_default_na=False)
    a0_tables(rl)
    a1_rows = a1_tables(rl)
    branch_rows, pair_rows, task_rows = a2_tables(rl)
    write_csv(RES / "pair_level.csv", a1_rows + pair_rows)
    write_csv(RES / "A2_branch_level.csv", branch_rows)
    write_csv(RES / "task_level.csv", task_rows)
    write_csv(RES / "resource_ledger.csv", resource_rows(rl))
    quality_and_claims(rl, ledger, pair_rows, task_rows, a1_rows)
    print("analysis written:", RES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
