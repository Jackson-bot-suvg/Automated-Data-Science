#!/usr/bin/env python3
"""Rebuild every lane-A paper table and figure from results/*.csv and raw run events.

Tables  : paper/tables/T-A0a_old_vs_current.csv, T-A0b_ablation.csv, T-A1_binding.csv,
          T-A2_context_x_decision.csv, T-A3_intake_only_vs_full.csv
Figures : paper/figures/F-A1_action_chain.{pdf,png}, F-A2_forest.{pdf,png},
          F-A3_same_prefix_branches.{pdf,png}, F-A4_location_x_increment.{pdf,png}
Index   : paper/asset_index.csv (asset -> source files -> command -> pairing level -> run ids)
Black-and-white figures (hatching/markers, no colour semantics).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PKG, WORK  # noqa: E402
from specs import A2_CONTEXTS  # noqa: E402

RES = PKG / "results"
TAB = PKG / "paper" / "tables"
FIG = PKG / "paper" / "figures"
CMD = "python scripts/make_paper_assets.py"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6, "savefig.dpi": 200})
CTX_SHORT = {
    "a_recoverable_intake_conflict": "(a) intake conflict",
    "b_unrecoverable_missing_target": "(b) target missing",
    "c_post_planner_fit_target_drift": "(c) post-planner fit drift",
    "d_predict_id_role_misalignment": "(d) ID/prediction misalignment",
}
TASK_SHORT = {"DEV-BIN-SPACESHIP": "Spaceship (BIN)", "DEV-MC-GHOULS": "Ghouls (MC)", "DEV-REG-HOUSE": "House (REG)"}
index_rows: list[dict] = []


def save(fig, name: str, sources: list[str], pairing: str, run_ids: list[str]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    index_rows.append({"asset": f"figures/{name}.pdf", "sources": "|".join(sources), "command": CMD, "normalisation": "counts / paired differences; no normalisation across tasks",
                       "pairing_level": pairing, "run_ids": "|".join(run_ids[:60])})


def table(df: pd.DataFrame, name: str, sources: list[str], pairing: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    rid_cols = [c for c in df.columns if "run" in c.lower() and "id" in c.lower()]
    rids = []
    for c in rid_cols:
        rids += [str(x) for x in df[c].tolist()]
    index_rows.append({"asset": f"tables/{name}.csv", "sources": "|".join(sources), "command": CMD, "normalisation": "none",
                       "pairing_level": pairing, "run_ids": "|".join(rids[:60])})


def events(run_id: str) -> list[dict]:
    out = []
    d = WORK / "runs" / run_id
    for name in ("prefix_events.jsonl", "events.jsonl"):
        p = d / name
        if p.exists():
            for line in p.read_text().splitlines():
                e = json.loads(line)
                e["_origin"] = "prefix" if name.startswith("prefix") else "own"
                out.append(e)
    return out


CASES = [
    ("positive_01_intake_blocks_missing_target", "A1__DEV-BIN-SPACESHIP__F01__F0_ORIGINAL__sb1101",
     "favourable: F0's intake gate stops an unrecoverable missing-target package before any fit (expected S3)."),
    ("positive_02_intake_blocks_wrong_hint", "A1__DEV-MC-GHOULS__F03__F0_ORIGINAL__sb1101",
     "favourable: F0's intake gate stops a misleading explicit target hint (expected S3)."),
    ("counter_01_post_intake_drift_passes", "A1__DEV-BIN-SPACESHIP__F05__F0_ORIGINAL__sb1101",
     "unfavourable: target_spec rewritten after intake; every later gate validates against the rewritten contract; wrong target delivered (S5)."),
    ("counter_02_legal_alias_synthesized_ids", "A1__DEV-REG-HOUSE__B01__F0_ORIGINAL__sb1101",
     "unfavourable: a legal output alias is not aligned; the alias column is filled with synthesized range ids and approved (S5)."),
    ("counter_03_permuted_pairing_passes", "A1__DEV-BIN-SPACESHIP__F10__F0_ORIGINAL__sb1101",
     "unfavourable: ID->prediction pairing permuted after predict(); columns/rows/nulls pass the model_building gate (S5)."),
    ("counter_04_right_stop_wrong_reason", "A0P__DEV-ML-STEEL__F01__F0_ORIGINAL__sb1101__r1",
     "mixed: Steel with all 7 targets removed; the reader's fuzzy hint fallback picks a feature as target and the gate stops on 'leakage' of that feature (S3 for the wrong reason)."),
]


def write_cases(rl: pd.DataFrame) -> None:
    out_dir = PKG / "paper" / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {"node_end", "gate_check", "gate_verdict", "injection", "fit_start", "fit_end", "predict", "serialize", "artifact_write", "run_end", "exception", "tool_error", "fork_snapshot", "fork_resume"}
    for name, rid, note in CASES:
        hit = rl[rl["run_id"] == rid]
        if hit.empty:
            continue
        r = hit.iloc[0]
        lines = [f"# {name}", "", f"- run_id: `{rid}`", f"- verdict of the oracle: **{r['terminal_S']}** — {r['terminal_subreason']}",
                 f"- why this case: {note}", f"- C_gold targets: see private oracle; C_read targets: `{r['C_read_targets']}`; C_used fit targets: `{r['C_used_fit_targets']}`",
                 f"- output provenance at serialization: `{r['C_used_output_provenance']}`", f"- raw trace: `logs/events/{rid}.jsonl`", "",
                 "| seq | event_id | event | actor | phase | key fields |", "|---|---|---|---|---|---|"]
        for e in events(rid):
            if e["event_type"] not in keep:
                continue
            if e["event_type"] == "node_end" and e.get("actor") not in {"reader", "planner"}:
                continue
            p = e.get("payload") or {}
            if e["event_type"] == "node_end" and e.get("actor") == "reader":
                c = p.get("contract_read_candidate") or {}
                key = f"C_read targets={c.get('target_columns')} family={c.get('task_family')} out={c.get('output_columns')} ids={c.get('id_columns')}"
            elif e["event_type"] == "node_end":
                key = f"planned_phase={p.get('planned_phase')} candidate_hash={str(e.get('candidate_hash'))[:12]}"
            elif e["event_type"] == "gate_check":
                key = "computed=" + "; ".join(f"[{f['class']}] {f['text'][:80]}" for f in p.get("computed_failures") or []) + f" | f0_status={e.get('verdict')}"
            elif e["event_type"] == "gate_verdict":
                key = f"verdict={e.get('verdict')} -> {e.get('next_action')}"
            elif e["event_type"] == "fit_start":
                key = f"y={p.get('y_columns')} family={p.get('task_family')} metric={p.get('primary_metric')} actual_fit_target_hash={str(e.get('actual_fit_target_hash'))[:12]}"
            elif e["event_type"] == "serialize":
                key = f"provenance={p.get('column_provenance')} mapping_digest={str(e.get('artifact_mapping_digest'))[:12]}"
            elif e["event_type"] == "injection":
                key = json.dumps({k: p.get(k) for k in ("template_id", "before_target_columns", "after_target_columns", "fixed_points") if k in p})
            elif e["event_type"] == "run_end":
                key = f"outcome={p.get('outcome')} review={p.get('latest_review_status')} completed={p.get('workflow_completed')} prediction={p.get('prediction_exists')}"
            else:
                key = json.dumps({k: p.get(k) for k in list(p)[:3]}, default=str)[:160]
            key = key.replace("|", "/")
            lines.append(f"| {e['monotonic_seq']} | `{e['event_id'][:12]}` | {e['event_type']} | {e.get('actor', '')} | {e.get('phase', '')} | {key} |")
        (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        index_rows.append({"asset": f"cases/{name}.md", "sources": f"logs/events/{rid}.jsonl", "command": CMD, "normalisation": "none", "pairing_level": "single run", "run_ids": rid})


def main() -> int:
    rl = pd.read_csv(RES / "run_level.csv", dtype=object, keep_default_na=False)
    # ---------------- T-A0 ----------------
    a0 = pd.read_csv(RES / "A0" / "A0_old_vs_current.csv")
    table(a0, "T-A0a_old_vs_current", ["results/A0/A0_old_vs_current.csv", "manuscript Table 2"], "per DEV task, 3 repeats")
    ab = pd.read_csv(RES / "A0" / "A0_old_vs_corrected_ablation.csv")
    table(ab, "T-A0b_old_vs_corrected_ablation", ["results/A0/A0_old_vs_corrected_ablation.csv", "manuscript Table 3"], "per method, 2 clean DEV tasks")

    # ---------------- T-A1 ----------------
    a1 = rl[rl["stage"] == "A1"].copy()
    cols = ["task_id", "template_id", "method_id", "run_id", "C_read_targets", "C_read_output_columns", "C_used_fit_targets", "C_used_family",
            "C_used_output_provenance", "fit_binding_category", "fault_detected_by_gate", "localization", "fatal_gate", "terminal_S", "terminal_subreason"]
    ta1 = a1[cols].sort_values(["task_id", "template_id", "method_id"])
    table(ta1, "T-A1_truth_intake_fit_output", ["results/run_level.csv (stage A1)", "events.jsonl fit_start/serialize"], "per run (F0 and M3 independent runs per task x probe)")

    # ---------------- F-A1: action chains from real run ids ----------------
    picks = []
    for rid in ["A1__DEV-BIN-SPACESHIP__C00__F0_ORIGINAL__sb1101", "A1__DEV-BIN-SPACESHIP__F03__F0_ORIGINAL__sb1101", "A1__DEV-BIN-SPACESHIP__F05__F0_ORIGINAL__sb1101",
                "A1__DEV-REG-HOUSE__B01__F0_ORIGINAL__sb1101", "A1__DEV-BIN-SPACESHIP__F10__F0_ORIGINAL__sb1101"]:
        if (WORK / "runs" / rid / "events.jsonl").exists():
            picks.append(rid)
    if picks:
        fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(picks) + 0.8))
        marker = {"contract": ("s", "C_read"), "gate_ok": ("o", "gate approved"), "gate_bad": ("x", "gate retry/fatal"), "fit": ("^", "fit(y)"),
                  "serialize": ("D", "serialize"), "inject": ("*", "injection"), "end": ("s", "run end")}
        used = set()
        for i, rid in enumerate(picks):
            evs = events(rid)
            row = rl[rl["run_id"] == rid].iloc[0]
            xs = []
            for e in evs:
                et = e["event_type"]
                key = None
                label = ""
                if et == "node_end" and e.get("actor") == "reader":
                    key, label = "contract", ",".join(((e.get("payload") or {}).get("contract_read_candidate") or {}).get("target_columns") or [])
                elif et == "gate_verdict":
                    key = "gate_ok" if e.get("verdict") == "approved" else "gate_bad"
                elif et == "fit_start":
                    key, label = "fit", ",".join((e.get("payload") or {}).get("y_columns") or [])
                elif et == "serialize":
                    prov = (e.get("payload") or {}).get("column_provenance") or {}
                    key, label = "serialize", ",".join(f"{k}:{v[:5]}" for k, v in prov.items())
                elif et == "injection":
                    key = "inject"
                elif et == "run_end":
                    key = "end"
                if key:
                    xs.append((e["monotonic_seq"], key, label))
            ax.plot([x[0] for x in xs], [i] * len(xs), color="0.7", lw=0.6, zorder=1)
            for seq, key, label in xs:
                m, lab = marker[key]
                ax.scatter([seq], [i], marker=m, s=28, facecolor="white" if key in {"gate_ok", "contract"} else "black", edgecolor="black", lw=0.7, zorder=2,
                           label=lab if key not in used else None)
                used.add(key)
                if label and key in {"contract", "fit"}:
                    ax.annotate(label[:18], (seq, i), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=5.5)
            ax.text(max(x[0] for x in xs) + 2, i, f"{row['terminal_S']}", va="center", fontsize=7, fontweight="bold")
        ax.set_yticks(range(len(picks)))
        ax.set_yticklabels([p.replace("A1__", "").replace("__F0_ORIGINAL__sb1101", "").replace("DEV-", "") for p in picks], fontsize=6.5)
        ax.set_xlabel("event sequence number (events.jsonl monotonic_seq)")
        ax.set_title("F-A1  Contract-to-action chains from real F0 runs (run_id on the left, oracle terminal on the right)", fontsize=8)
        ax.legend(ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, "F-A1_action_chain", ["runs/*/events.jsonl", "results/run_level.csv"], "single runs", picks)

    # ---------------- T-A2 ----------------
    br = pd.read_csv(RES / "A2_branch_level.csv", dtype=object, keep_default_na=False) if (RES / "A2_branch_level.csv").exists() else pd.DataFrame()
    if not br.empty:
        order = ["M3_SHADOW", "M5_BLOCK_ONLY", "G_GENERIC_RETRY", "F0_ORIGINAL", "M4_END_ONLY", "M2_INTAKE_ONLY_STRONG"]
        rows = []
        for ctx in A2_CONTEXTS:
            for m in order:
                g = br[(br["context"] == ctx) & (br["method_id"] == m)]
                if g.empty:
                    rows.append({"context": ctx, "method_id": m, "n": 0, "terminals": "N/A (pre-registered)" if m == "G_GENERIC_RETRY" and ctx[0] in "ab" else "not run"})
                    continue
                counts = g["terminal_S"].value_counts().to_dict()
                rows.append({"context": ctx, "method_id": m, "n": len(g),
                             "next_action_after_gate": "|".join(sorted(set(g["next_action"]))),
                             "wrong_fit_after_fork": int(g["wrong_fits_after_fork"].astype(float).gt(0).sum()),
                             "wrong_file_delivered": int((g["wrong_file_delivered"] == "True").sum()),
                             "S1": counts.get("S1", 0), "S2": counts.get("S2", 0), "S3": counts.get("S3", 0), "S4": counts.get("S4", 0),
                             "S5": counts.get("S5", 0), "S6": counts.get("S6", 0), "S7": counts.get("S7", 0), "S8": counts.get("S8", 0),
                             "mean_api_calls_after_fork": round(g["api_calls_after_fork"].astype(float).mean(), 2),
                             "mean_wrong_training_s_after_fork": round(g["wrong_training_seconds_after_fork"].astype(float).mean(), 2),
                             "strict_common_prefix": f"{(g['strict_common_prefix'] == 'True').sum()}/{len(g)}",
                             "run_ids": "|".join(g["run_id"])})
        table(pd.DataFrame(rows), "T-A2_context_x_decision", ["results/A2_branch_level.csv"], "same-prefix branches (24 DEV prefixes)")

        # ---------------- F-A2 forest ----------------
        tl = pd.read_csv(RES / "task_level.csv", dtype=object, keep_default_na=False)
        pr = pd.read_csv(RES / "pair_level.csv", dtype=object, keep_default_na=False)
        metrics = [("d_unsafe_delivery_S5S6", "Δ unsafe delivery (S5+S6)"), ("d_correct_delivery_S1S2", "Δ correct delivery (S1+S2)"), ("d_justified_stop_S3", "Δ justified stop (S3)")]
        contrasts = ["F0-M3", "F0-M5", "F0-M4", "F0-M2"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(7.4, 4.2), sharey=True)
        rows_present = [(c, x) for c in contrasts for x in A2_CONTEXTS
                        if not tl[(tl["contrast"] == c) & (tl["context"] == x)].empty]
        ylabels = [f"{c} {CTX_SHORT[x]}" for c, x in rows_present]
        for ax, (metric, title) in zip(axes, metrics):
            y = 0
            for contrast, ctx in rows_present:
                if True:
                    t = tl[(tl["contrast"] == contrast) & (tl["context"] == ctx) & (tl["metric"] == metric)]
                    if not t.empty:
                        r = t.iloc[0]
                        per = json.loads(r["per_task"])
                        for k, v in per.items():
                            ax.scatter(v, y + {"DEV-BIN-SPACESHIP": -0.18, "DEV-MC-GHOULS": 0, "DEV-REG-HOUSE": 0.18}.get(k, 0), s=9,
                                       marker={"DEV-BIN-SPACESHIP": "o", "DEV-MC-GHOULS": "s", "DEV-REG-HOUSE": "^"}.get(k, "o"), facecolor="white", edgecolor="black", lw=0.6)
                        mean = float(r["task_level_mean"])
                        lo = float(r["boot95_low"]) if r["boot95_low"] not in {"NA", ""} else mean
                        hi = float(r["boot95_high"]) if r["boot95_high"] not in {"NA", ""} else mean
                        ax.plot([lo, hi], [y, y], color="black", lw=1)
                        ax.scatter([mean], [y], marker="|", s=60, color="black")
                    y += 1
            ax.axvline(0, color="0.5", lw=0.6, ls="--")
            ax.set_title(title, fontsize=7.5)
            ax.set_xlim(-1.15, 1.15)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_yticks(range(len(ylabels)))
        axes[0].set_yticklabels(ylabels, fontsize=5.6)
        axes[0].invert_yaxis()
        fig.suptitle("F-A2  Task-level paired effects per context (DEV pilot, 3 tasks x 2 seed blocks). Bar: task-mean with task-bootstrap 95% (unstable, n=3);\nmarkers: per task (o Spaceship, □ Ghouls, △ House). Negative Δ unsafe = fewer unsafe deliveries for the first method.", fontsize=7)
        save(fig, "F-A2_forest", ["results/task_level.csv", "results/pair_level.csv"], "task-level mean of same-prefix paired differences", pr["run_a"].tolist() + pr["run_b"].tolist())

        # ---------------- F-A3: same-prefix five branches ----------------
        pids = sorted(br["pair_id"].unique())
        candidates = [p for p in pids if "a_recoverable" in p and "SPACESHIP" in p and p.endswith("sb1101")][:1] + [p for p in pids if "c_post" in p and "HOUSE" in p and p.endswith("sb1101")][:1]
        if candidates:
            fig, axes = plt.subplots(1, len(candidates[:2]), figsize=(7.4, 2.8))
            axes = np.atleast_1d(axes)
            for ax, pid in zip(axes, candidates[:2]):
                g = br[br["pair_id"] == pid]
                rid_list = []
                xmax_seen = 0
                for i, m in enumerate(order):
                    hit = g[g["method_id"] == m]
                    if hit.empty:
                        ax.text(1, i, "N/A (pre-registered)" if m == "G_GENERIC_RETRY" else "not in plan", va="center", fontsize=6, color="0.4")
                        continue
                    rid = hit.iloc[0]["run_id"]
                    rid_list.append(rid)
                    evs = [e for e in events(rid) if e["_origin"] == "own"]
                    pts = []
                    for e in evs:
                        et = e["event_type"]
                        if et == "gate_verdict":
                            pts.append((e["monotonic_seq"], "o" if e.get("verdict") == "approved" else ("x" if e.get("verdict") == "fatal" else "v")))
                        elif et == "fit_start":
                            pts.append((e["monotonic_seq"], "^"))
                        elif et == "artifact_write":
                            pts.append((e["monotonic_seq"], "D"))
                    if not pts:
                        continue
                    ax.plot([p[0] for p in pts], [i] * len(pts), color="0.6", lw=0.6)
                    for s, mk in pts:
                        ax.scatter([s], [i], marker=mk, s=22, facecolor="white" if mk in {"o", "D"} else "black", edgecolor="black", lw=0.6)
                    ax.annotate(hit.iloc[0]["terminal_S"], (max(p[0] for p in pts), i), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7, fontweight="bold")
                    xmax_seen = max(xmax_seen, max(p[0] for p in pts))
                ax.set_xlim(0, max(xmax_seen * 1.15, 20))
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels([m.split("_")[0] for m in order], fontsize=6.5)
                ax.set_title(pid.replace("A2P__", "").replace("DEV-", ""), fontsize=6.5)
                ax.set_xlabel("event seq after fork (own events)")
                ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle("F-A3  Same-prefix branches from identical snapshots (o approved, v retry, x fatal, ^ fit, ◇ file write; terminal at right)", fontsize=7)
            save(fig, "F-A3_same_prefix_branches", ["runs/A2__*/events.jsonl", "results/A2_branch_level.csv"], "one prefix, all branches", list(br[br["pair_id"].isin(candidates[:2])]["run_id"]))

        # ---------------- T-A3 and F-A4 ----------------
        a3 = pr[pr["contrast"] == "F0-M2"] if "contrast" in pr.columns else pd.DataFrame()
        if not a3.empty:
            t3 = a3[["pair_id", "context", "task_id", "seed_block", "terminal_a", "terminal_b", "d_unsafe_delivery_S5S6", "d_correct_delivery_S1S2", "d_justified_stop_S3", "strict_common_prefix", "run_a", "run_b"]].rename(columns={"terminal_a": "F0_terminal", "terminal_b": "M2_terminal"})
            table(t3, "T-A3_intake_only_vs_full", ["results/pair_level.csv (contrast F0-M2)"], "same-prefix F0 vs M2 pairs")
            fig, ax = plt.subplots(figsize=(4.6, 2.6))
            terms = ["S3", "S5", "S6", "S7"]
            hatches = {"S3": "", "S5": "////", "S6": "....", "S7": "xxxx"}
            facec = {"S3": "white", "S5": "0.55", "S6": "white", "S7": "white"}
            xt, xl = [], []
            for j, ctx in enumerate(["a_recoverable_intake_conflict", "c_post_planner_fit_target_drift"]):
                g = a3[a3["context"] == ctx]
                for k, (col, lab) in enumerate([("terminal_a", "F0"), ("terminal_b", "M2")]):
                    x = j * 3 + k
                    bottom = 0
                    for tname in terms:
                        n = int((g[col] == tname).sum())
                        if n:
                            ax.bar(x, n, bottom=bottom, color=facec[tname], edgecolor="black", hatch=hatches[tname], width=0.75,
                                   label=tname if (j == 0 and k == 0) or tname not in [h.get_label() for h in ax.patches] else None)
                            ax.text(x, bottom + n / 2, f"{tname}×{n}", ha="center", va="center", fontsize=6, backgroundcolor="white")
                            bottom += n
                    xt.append(x)
                    xl.append(f"{lab}")
                same = int((g["terminal_a"] == g["terminal_b"]).sum())
                ax.text(j * 3 + 0.5, 6.4, f"identical terminal in {same}/{len(g)} pairs", ha="center", fontsize=6.5)
            ax.set_xticks(xt)
            ax.set_xticklabels(xl, fontsize=7)
            ax.set_ylim(0, 7.2)
            ax.set_ylabel("runs (6 same-prefix pairs per context)")
            ax.text(0.5, -1.25, "fault at intake (a)", ha="center", fontsize=7)
            ax.text(3.5, -1.25, "fault after intake (c)", ha="center", fontsize=7)
            ax.set_title("F-A4  Fault location × F0 vs intake-only M2 (same snapshots)", fontsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)
            save(fig, "F-A4_location_x_increment", ["results/pair_level.csv"], "same-prefix F0 vs M2", a3["run_a"].tolist() + a3["run_b"].tolist())

    write_cases(rl)
    with open(PKG / "paper" / "asset_index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["asset", "sources", "command", "normalisation", "pairing_level", "run_ids"])
        w.writeheader()
        w.writerows(index_rows)
    print("assets:", len(index_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
