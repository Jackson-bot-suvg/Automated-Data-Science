#!/usr/bin/env python3
"""Independent offline evaluation: every launched run -> exactly one row of results/run_level.csv.

Reads only: run directories (events, run_result.json, delivered file, lane-private hook
frames), cases/manifest.csv, the private C_gold files and private test labels.  It never
reads agent self-assessments.  Prefix runs that reached their fork point are written with
terminal_S = PREFIX_FORKED (their trajectories are scored through their continuation
branches); a prefix that ended before its fork point is a terminal run in its own right.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import BASE_COMMIT, DATA_ROOT, PKG, PRIVATE, WORK  # noqa: E402
from oracle import ORACLE_VERSION  # noqa: E402
from oracle.evaluate_core import evaluate_run  # noqa: E402
from specs import DEV_TASKS, PROBE_PARAMS, TEMPLATES  # noqa: E402

RESULTS = PKG / "results"
OUTPUTS = PKG / "outputs"


def gold_for(task_id: str, template: str, case_id: str) -> tuple[dict, Path | None, pd.DataFrame | None]:
    spec = DEV_TASKS[task_id]
    if case_id.endswith(":ORIG"):
        # A0 manuscript reproduction on the original package: no private labels exist
        sub_cols = pd.read_csv(DATA_ROOT / spec["data_dir"] / "sample_submission.csv", nrows=0).columns.tolist()
        gold = {
            "task_id": task_id,
            "template_id": "C00",
            "C_gold_version": f"{ORACLE_VERSION}/ORIG",
            "target_columns": spec["target_columns"],
            "task_family": spec["family"],
            "requested_metric": spec["metric_hint"],
            "official_metric": spec["official_metric"],
            "id_columns": spec["id_columns"],
            "output_columns": spec["target_columns"],
            "output_mapping": {t: t for t in spec["target_columns"]},
            "submission_columns": sub_cols,
            "label_domain": spec.get("label_domain"),
        }
        return gold, DATA_ROOT / spec["data_dir"], None
    gold = json.loads((PRIVATE / "oracle_gold" / f"{task_id}__{template}.json").read_text())
    pkg_template = "C00" if template in {"F05", "F10"} else template
    labels = pd.read_csv(PRIVATE / "private_test_labels" / f"{task_id}.csv", dtype=str, keep_default_na=False)
    return gold, WORK / "packages" / task_id / pkg_template, labels


RUN_LEVEL_FIELDS = [
    "run_id", "plan_cell_id", "pair_id", "stage", "task_id", "source_group_id", "case_id", "case_family", "method_id",
    "implementation_sha", "split_hash", "oracle_version", "prompt_sha", "config_sha", "seed_block", "host_id", "model_id",
    "started_at", "finished_at", "terminal_S", "terminal_subreason", "semantic_correct", "artifact_valid",
    "actual_fit_target_hash", "actual_metric", "output_mapping_valid", "rejected_valid_case", "wall_seconds",
    "wrong_training_seconds", "api_calls", "tokens_in_if_observed", "tokens_out_if_observed", "measured_cost_if_observed",
    "artifact_sha256", "event_trace_path", "exception_path", "missingness_reason",
    # lane-A specific
    "template_id", "condition", "information_level", "safe_terminal", "delivered", "C_read_targets", "C_read_family",
    "C_read_output_columns", "C_read_id_columns", "C_used_fit_targets", "C_used_family", "C_used_output_provenance",
    "C_read_matches_gold", "fit_target_matches_gold", "task_family_matches_gold", "metric_aligned", "fit_binding_category",
    "n_fits", "n_gate_checks", "n_enforced_retry", "fatal_gate", "fatal_due_to_S", "computed_S_failures", "fault_detected_by_gate",
    "localization", "tool_errors", "invalid_reason", "mapping_reason", "score_requested_metric", "score_official_metric",
    "api_calls_own", "api_errors", "returned_model_ids", "prefix_wall_seconds", "injection_applied", "injection_verified",
    "fork_state_hash_match", "fork_candidate_hash_match", "fork_rng_match", "outcome_raw", "n_events",
    "fork_gate_verdict", "fork_gate_next_action", "fork_gate_f0_counterfactual", "wrong_fits_after_fork",
    "wrong_training_seconds_after_fork", "fits_after_fork",
]


def _manifest() -> dict[str, dict]:
    with open(PKG / "cases" / "manifest.csv", encoding="utf-8") as fh:
        return {r["case_id"]: r for r in csv.DictReader(fh)}


def _split_hashes() -> dict[str, str]:
    with open(PKG / "protocol" / "split_registry.csv", encoding="utf-8") as fh:
        return {r["task_id"]: r["split_hash"] for r in csv.DictReader(fh)}


def main() -> int:
    sys.path.insert(0, str(PKG / "cases" / "mutation_specs"))
    import mutators as M

    manifest = _manifest()
    split_hash = _split_hashes()
    impl_sha = json.loads((PKG / "logs" / "implementation_sha.json").read_text()) if (PKG / "logs" / "implementation_sha.json").exists() else {}
    final = "--final" in sys.argv
    rows, art_rows = [], []
    runs_dir = WORK / "runs"
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        spec_path = WORK / "specs" / f"{run_dir.name}.json"
        if not spec_path.exists():
            continue
        if not (run_dir / "run_result.json").exists() and not final:
            continue  # still running; with --final a missing result is scored S8/infrastructure
        spec = json.loads(spec_path.read_text())
        task_id, template = spec["task_id"], spec.get("template_id", "C00")
        case_id = spec.get("case_id", "")
        gold, pkg, labels = gold_for(task_id, template, case_id)
        mrow = manifest.get(case_id, {})
        tmpl = TEMPLATES.get(template, TEMPLATES["C00"])
        wrong = [PROBE_PARAMS.get(task_id, {}).get("drift_target")] if template == "F05" else (
            [PROBE_PARAMS.get(task_id, {}).get("wrong_hint")] if template == "F03" else None)
        r = evaluate_run(run_dir, gold=gold, template=template, package_dir=pkg, labels=labels,
                         safe_terminals=tmpl["safe_terminals"], fault_fields=tmpl["target_fault_fields"], wrong_target=wrong)
        res = json.loads((run_dir / "run_result.json").read_text()) if (run_dir / "run_result.json").exists() else {}
        if r["terminal_S"] == "PREFIX":
            r["terminal_S"] = "PREFIX_FORKED"
        if spec.get("observers") is False:
            # hook-equivalence reference: actions are deliberately unobserved.  Semantic status is
            # inferred only from byte identity with the observed F0 replay of the same deterministic run.
            ref = WORK / "runs" / run_dir.name.replace("F0_UNOBSERVED", "F0_ORIGINAL") / "run_result.json"
            ref_sha = json.loads(ref.read_text()).get("prediction_sha256") if ref.exists() else None
            same = bool(ref_sha) and ref_sha == res.get("prediction_sha256")
            r["semantic_correct"] = "NA(actions unobserved by design)"
            if res.get("workflow_completed") and res.get("prediction_exists") and same:
                r["terminal_S"] = "S1"
                r["terminal_subreason"] = "unobserved reference: delivered file byte-identical to observed F0 replay " + ref.parent.name
            else:
                r["terminal_S"] = "S7" if not res.get("prediction_exists") else "S6"
                r["terminal_subreason"] = "unobserved reference differs from observed replay (see version_diff.csv)"
        evs = []
        for name in ("prefix_events.jsonl", "events.jsonl"):
            p = run_dir / name
            if p.exists():
                evs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        start = next((e for e in evs if e["event_type"] == "run_start"), {})
        own_start = next((json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if '"run_start"' in l), {}) if (run_dir / "events.jsonl").exists() else {}
        own_end = next((json.loads(l) for l in reversed((run_dir / "events.jsonl").read_text().splitlines()) if '"run_end"' in l), {}) if (run_dir / "events.jsonl").exists() else {}
        resume = next((e for e in evs if e["event_type"] == "fork_resume"), None)
        inj_ok = None
        if template in {"F05", "F10"}:
            v = M.verify_runtime_injection(evs, template, wrong_target=[PROBE_PARAMS[task_id]["drift_target"]] if template == "F05" else None)
            inj_ok = v["ok"]
        llm_prompts = sorted({(e.get("payload") or {}).get("prompt_sha256", "") for e in evs if e["event_type"] == "llm_call"} - {""})
        row = {
            "run_id": run_dir.name,
            "plan_cell_id": spec.get("plan_cell_id", run_dir.name),
            "pair_id": spec.get("pair_id", ""),
            "stage": spec.get("stage", ""),
            "task_id": task_id,
            "source_group_id": f"SRC-{DEV_TASKS[task_id]['data_dir']}",
            "case_id": case_id,
            "case_family": tmpl["case_family"],
            "method_id": spec["method_id"] if spec.get("mode") != "prefix" else "PREFIX(F0)",
            "implementation_sha": impl_sha.get("adapter_tree_sha256", "") + "@" + BASE_COMMIT[:12],
            "split_hash": "" if case_id.endswith(":ORIG") else split_hash.get(task_id, ""),
            "oracle_version": ORACLE_VERSION,
            "prompt_sha": __import__("hashlib").sha256("|".join(llm_prompts).encode()).hexdigest() if llm_prompts else "NA(no LLM prompts observed)",
            "config_sha": __import__("hashlib").sha256(spec_path.read_bytes()).hexdigest(),
            "seed_block": spec.get("seed_block"),
            "host_id": spec.get("host_id", ""),
            "model_id": spec.get("model_id", "") if spec.get("llm_mode", "live") == "live" else "NA(LLM-off deterministic replay)",
            "started_at": own_start.get("timestamp", ""),
            "finished_at": own_end.get("timestamp", ""),
            "wall_seconds": r.get("wall_seconds_own"),
            "api_calls": r.get("api_calls_total"),
            "measured_cost_if_observed": "NA(provider price not frozen; tokens reported)",
            "artifact_sha256": res.get("prediction_sha256", "") or "NA(no artifact)",
            "event_trace_path": str((run_dir / "events.jsonl").relative_to(WORK)),
            "exception_path": str((run_dir / "errors").relative_to(WORK)) if (run_dir / "errors").exists() else "",
            "missingness_reason": "" if r.get("tokens_in_if_observed") is not None else "token usage not observed (LLM-off or no LLM call)",
            "information_level": tmpl["information_level"],
            "injection_verified": inj_ok,
            "fork_state_hash_match": (resume or {}).get("payload", {}).get("state_hash_match") if resume else None,
            "fork_candidate_hash_match": (resume or {}).get("payload", {}).get("candidate_hash_match") if resume else None,
            "fork_rng_match": (resume or {}).get("payload", {}).get("rng_match") if resume else None,
            "condition": spec.get("condition", ""),
            "template_id": template,
        }
        for k, v in r.items():
            row.setdefault(k, v)
        rows.append(row)
        art_rows.append({
            "run_id": run_dir.name,
            "artifact_path": res.get("prediction_path", ""),
            "artifact_sha256": res.get("prediction_sha256", ""),
            "delivered": r["delivered"],
            "artifact_valid": r["artifact_valid"],
            "terminal_S": row["terminal_S"],
            "invalid_or_missing_reason": r.get("invalid_reason") or ("" if res.get("prediction_exists") else "no prediction artifact"),
        })
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "run_level.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_LEVEL_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("NA" if row.get(k) is None else row.get(k)) for k in RUN_LEVEL_FIELDS})
    with open(OUTPUTS / "artifact_index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(art_rows[0].keys()) if art_rows else ["run_id"])
        w.writeheader()
        for row in art_rows:
            w.writerow(row)
    from collections import Counter

    print("runs evaluated:", len(rows))
    print(Counter((r["stage"], r["terminal_S"]) for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
