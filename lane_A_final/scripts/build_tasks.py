#!/usr/bin/env python3
"""Build DEV task registry, private splits, agent-visible packages and the case manifest.

Outputs
  protocol/task_registry_public.csv   DEV rows (full), CONF rows ([待冻结]), TR rows (id + sealed hash only)
  protocol/split_registry.csv         split policy, seed 1729, split hashes
  protocol/splits/<task_id>.json      fit / validation / private_test row indices (no labels)
  <WORK>/packages/<task_id>/<template>/  agent-visible packages (pilot tasks + Steel for A0 probes)
  <PRIVATE>/oracle_gold/<task_id>.json  C_gold (evaluator only)
  <PRIVATE>/private_test_labels/<task_id>.csv  (evaluator only; never packaged)
  cases/manifest.csv                  one row per (task, template) case

Idempotent: existing packages are verified, never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_ROOT, PKG, PRIVATE, SPLIT_SEED, WORK  # noqa: E402
import mutators as M  # noqa: E402
from specs import DEV_TASKS, PILOT_TASKS, PROBE_PARAMS, SPECS_VERSION, TEMPLATES  # noqa: E402

PROTOCOL = PKG / "protocol"
ORACLE_VERSION = "laneA-oracle-1.0"
A0_PROBE_TASK = "DEV-ML-STEEL"


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def source_hashes(task_id: str, spec: dict) -> dict[str, str]:
    src = DATA_ROOT / spec["data_dir"]
    return {p.name: M.sha256_file(p) for p in sorted(src.iterdir()) if p.is_file()}


def make_split(task_id: str, spec: dict) -> dict:
    """60% fit / 20% validation / 20% private test, seed 1729, stratified for single-target classification."""

    train = pd.read_csv(DATA_ROOT / spec["data_dir"] / "train.csv")
    idx = np.arange(len(train))
    strat = None
    if spec.get("stratify") and len(spec["target_columns"]) == 1:
        strat = train[spec["target_columns"][0]].astype(str).to_numpy()
    fit_idx, rest_idx = train_test_split(idx, test_size=0.4, random_state=SPLIT_SEED, stratify=strat)
    strat_rest = strat[rest_idx] if strat is not None else None
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=SPLIT_SEED, stratify=strat_rest)
    split = {
        "task_id": task_id,
        "policy": "random 60/20/20 (stratified by target for single-target classification); sklearn train_test_split",
        "split_seed": SPLIT_SEED,
        "source_file": "train.csv",
        "source_sha256": M.sha256_file(DATA_ROOT / spec["data_dir"] / "train.csv"),
        "fit": sorted(int(i) for i in fit_idx),
        "validation": sorted(int(i) for i in val_idx),
        "private_test": sorted(int(i) for i in test_idx),
    }
    split["split_hash"] = sha_json({k: split[k] for k in ("fit", "validation", "private_test", "split_seed", "source_sha256")})
    return split


def build_c00(task_id: str, spec: dict, split: dict) -> Path:
    dst = WORK / "packages" / task_id / "C00"
    if dst.exists():
        return dst
    src = DATA_ROOT / spec["data_dir"]
    train = pd.read_csv(src / "train.csv", dtype=str, keep_default_na=False)
    sub0 = pd.read_csv(src / "sample_submission.csv", dtype=str, keep_default_na=False)
    targets = spec["target_columns"]
    ids = spec["id_columns"]
    fit = train.iloc[split["fit"]]
    ptest = train.iloc[split["private_test"]]
    dst.mkdir(parents=True)
    fit.to_csv(dst / "train.csv", index=False)
    ptest.drop(columns=targets).to_csv(dst / "test.csv", index=False)
    sub = pd.DataFrame({col: ptest[col].to_numpy() for col in ids})
    for col in sub0.columns:
        if col in ids:
            continue
        sub[col] = sub0[col].iloc[0]
    sub = sub[list(sub0.columns)]
    sub.to_csv(dst / "sample_submission.csv", index=False)
    for name in M.OVERVIEW_FILES:
        if (src / name).exists():
            (dst / name).write_bytes((src / name).read_bytes())
    # evaluator-only material
    labels_dir = PRIVATE / "private_test_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    ptest[ids + targets].to_csv(labels_dir / f"{task_id}.csv", index=False)
    val_dir = PRIVATE / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    train.iloc[split["validation"]].to_csv(val_dir / f"{task_id}.csv", index=False)
    return dst


def gold_contract(task_id: str, spec: dict, template: str, alias_map: dict | None = None) -> dict:
    out_cols = list(spec["target_columns"])
    if alias_map:
        out_cols = [alias_map[c] for c in spec["target_columns"]]
    return {
        "task_id": task_id,
        "template_id": template,
        "C_gold_version": f"{ORACLE_VERSION}/{SPECS_VERSION}",
        "target_columns": list(spec["target_columns"]),
        "task_family": spec["family"],
        "requested_metric": spec["metric_hint"],
        "official_metric": spec["official_metric"],
        "id_columns": list(spec["id_columns"]),
        "output_columns": out_cols,
        "output_mapping": {t: o for t, o in zip(spec["target_columns"], out_cols)},
        "submission_columns": list(spec["id_columns"]) + out_cols,
        "label_domain": spec.get("label_domain"),
    }


def main() -> int:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    (PROTOCOL / "splits").mkdir(exist_ok=True)
    (PRIVATE / "oracle_gold").mkdir(parents=True, exist_ok=True)

    registry_rows = []
    split_rows = []
    splits = {}
    for task_id, spec in DEV_TASKS.items():
        split_path = PROTOCOL / "splits" / f"{task_id}.json"
        if split_path.exists():
            split = json.loads(split_path.read_text())
        else:
            split = make_split(task_id, spec)
            split_path.write_text(json.dumps(split))
        splits[task_id] = split
        hashes = source_hashes(task_id, spec)
        registry_rows.append(
            {
                "task_id": task_id,
                "split_tier": "DEV",
                "source_group_id": f"SRC-{spec['data_dir']}",
                "source_name": spec["data_dir"],
                "source_url": spec["source_url"],
                "license": "Kaggle competition rules (redistribution restricted; obtain from source_url)",
                "original_version": "base repo b9dbebe data/" + spec["data_dir"],
                "source_sha256": sha_json(hashes),
                "task_family": spec["family"],
                "target_columns": "|".join(spec["target_columns"]),
                "target_semantics": spec["q"],
                "primary_metric": spec["metric_hint"],
                "metric_direction": "maximize (negated error; i.e. minimize RMSE)" if spec["metric_hint"].startswith("neg_") else "maximize",
                "id_columns": "|".join(spec["id_columns"]),
                "output_columns": "|".join(spec["target_columns"]),
                "output_mapping": "identity",
                "valid_aliases": "B01 alias packages only (see cases/manifest.csv)",
                "data_split_policy": split["policy"],
                "split_seed": SPLIT_SEED,
                "split_hash": split["split_hash"],
                "oracle_version": ORACLE_VERSION,
                "applicability_notes": spec.get("applicability_note", "")
                + (" official_metric=" + spec["official_metric"] + "; manuscript/requested metric=" + spec["metric_hint"]),
                "owner_signed_at": "[待填]",
            }
        )
        split_rows.append(
            {
                "task_id": task_id,
                "split_tier": "DEV",
                "split_id": f"{task_id}/seed{SPLIT_SEED}",
                "policy": split["policy"],
                "split_seed": SPLIT_SEED,
                "n_fit": len(split["fit"]),
                "n_validation": len(split["validation"]),
                "n_private_test": len(split["private_test"]),
                "source_sha256": split["source_sha256"],
                "split_hash": split["split_hash"],
                "private_test_label_sha256": "",
                "notes": "agent-visible train = fit rows only (internal CV stays inside fit); validation rows held by the evaluator, unused by RTS-MA-DS (no external selection API)",
            }
        )

    # CONF / TRANSFER placeholders: never invented (protocol §2)
    for fam in ("BIN", "MC", "ML", "REG"):
        for k in (1, 2, 3):
            registry_rows.append({"task_id": f"CONF-{fam}-0{k}", "split_tier": "CONF", "source_name": "[具体数据待冻结]", "owner_signed_at": "[待填]"})
    for fam in ("BIN", "MC", "ML", "REG"):
        registry_rows.append({"task_id": f"TR-{fam}-01", "split_tier": "TRANSFER", "source_name": "[封存名称]", "applicability_notes": "sealed_manifest_sha256=[待填]"})

    manifest_rows = []
    build_for = [t for t in PILOT_TASKS.values() if t] + [A0_PROBE_TASK]
    for task_id in build_for:
        spec = DEV_TASKS[task_id]
        split = splits[task_id]
        c00 = build_c00(task_id, spec, split)
        params = PROBE_PARAMS[task_id]
        label_path = PRIVATE / "private_test_labels" / f"{task_id}.csv"
        for row in split_rows:
            if row["task_id"] == task_id:
                row["private_test_label_sha256"] = M.sha256_file(label_path)
        base_sha = M.package_digest(c00)
        templates = ["C00", "B01", "F03", "F01", "F05", "F10"] if task_id != A0_PROBE_TASK else ["C00", "B01", "F03", "F01", "F05"]
        for template in templates:
            hint = spec["hint_target"]
            pkg = c00
            changed: list[str] = []
            mparams: dict = {}
            amap = None
            verify = {"ok": True, "errors": [], "kind": "none"}
            if template == "B01":
                pkg = WORK / "packages" / task_id / "B01"
                amap = M.alias_map(spec["target_columns"], params["alias"])
                if not pkg.exists():
                    info = M.make_b01(c00, pkg, target_columns=spec["target_columns"], alias=params["alias"])
                    changed, mparams = info["changed_paths"], info["params"]
                else:
                    changed, mparams = ["sample_submission.csv"] + [n for n in M.OVERVIEW_FILES if (pkg / n).exists()], {"alias_map": amap}
                verify = {**M.verify_b01(c00, pkg, target_columns=spec["target_columns"], alias=params["alias"]), "kind": "file"}
            elif template == "F01":
                pkg = WORK / "packages" / task_id / "F01"
                if not pkg.exists():
                    info = M.make_f01(c00, pkg, target_columns=spec["target_columns"])
                    changed, mparams = info["changed_paths"], info["params"]
                else:
                    changed, mparams = ["train.csv"], {"dropped": spec["target_columns"]}
                verify = {**M.verify_f01(c00, pkg, target_columns=spec["target_columns"]), "kind": "file"}
            elif template == "F03":
                pkg = WORK / "packages" / task_id / "F03"
                hint = params["wrong_hint"]
                if not pkg.exists():
                    info = M.make_f03(c00, pkg, wrong_hint=hint)
                    changed, mparams = info["changed_paths"], info["params"]
                else:
                    changed, mparams = ["H.target_hint"], {"target_hint": hint}
                verify = {**M.verify_f03(c00, pkg, wrong_hint=hint, target_columns=spec["target_columns"]), "kind": "file+hint"}
            elif template == "F05":
                changed, mparams = ["runtime:target_spec.target_columns"], {"wrong_target_columns": [params["drift_target"]]}
                verify = {"ok": None, "errors": [], "kind": "runtime(verified per run from events)"}
            elif template == "F10":
                changed, mparams = ["runtime:submission value rows"], {"permutation_seed": "seed_block"}
                verify = {"ok": None, "errors": [], "kind": "runtime(verified per run from events)"}
            tmpl = TEMPLATES[template]
            gold = gold_contract(task_id, spec, template, amap)
            gold_path = PRIVATE / "oracle_gold" / f"{task_id}__{template}.json"
            gold_path.write_text(json.dumps(gold, indent=2, sort_keys=True))
            visible_hash = M.package_digest(pkg, q=spec["q"], hint=hint, metric_hint=spec["metric_hint"])
            manifest_rows.append(
                {
                    "case_id": f"{task_id}:{template}",
                    "task_id": task_id,
                    "template_id": template,
                    "condition": tmpl["condition"],
                    "injection_stage": tmpl["injection_stage"],
                    "base_sha256": base_sha,
                    "package_path": str(pkg.relative_to(WORK)),
                    "task_text_q": spec["q"],
                    "target_hint_H": hint,
                    "metric_hint_H": spec["metric_hint"],
                    "changed_paths": "|".join(changed),
                    "mutation_params": json.dumps(mparams, sort_keys=True),
                    "mutation_sha256": sha_json({"template": template, "params": mparams, "package": M.package_digest(pkg)}),
                    "visible_evidence_hash": visible_hash,
                    "C_gold_version": gold["C_gold_version"],
                    "C_gold_sha256": M.sha256_file(gold_path),
                    "truth_status": tmpl["truth_status"],
                    "information_level": tmpl["information_level"],
                    "recoverable": tmpl["recoverable"],
                    "expected_safe_action": tmpl["expected_safe_action"],
                    "safe_terminals": "|".join(tmpl["safe_terminals"]),
                    "target_fault_fields": "|".join(tmpl["target_fault_fields"]),
                    "expected_detection_stage": tmpl["expected_detection_stage"],
                    "applicability": "A0-only(manuscript ML representative; not genuine multilabel)" if task_id == A0_PROBE_TASK else "A1/A2/A3",
                    "pre_registered_exclusion_reason": "",
                    "mutation_verified": verify["ok"],
                    "verification_kind": verify["kind"],
                    "verification_errors": "; ".join(verify["errors"]),
                    "specs_version": SPECS_VERSION,
                    "mutators_version": M.MUTATORS_VERSION,
                }
            )

    # ML family of the DEV pilot: no genuine multilabel DEV task -> pre-registered N/A
    for template in ["C00", "B01", "F03", "F01", "F05", "F10"]:
        manifest_rows.append(
            {
                "case_id": f"DEV-ML-(none):{template}",
                "task_id": "DEV-ML-(none)",
                "template_id": template,
                "condition": TEMPLATES[template]["condition"],
                "applicability": "N/A",
                "pre_registered_exclusion_reason": "no genuine multilabel DEV task (Steel Plate co-occurrence 0.11%); ML family requires CONF-ML-01..03 (not frozen)",
            }
        )

    def write_csv(path: Path, rows: list[dict]) -> None:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(PROTOCOL / "task_registry_public.csv", registry_rows)
    write_csv(PROTOCOL / "split_registry.csv", split_rows)
    (PKG / "cases").mkdir(exist_ok=True)
    write_csv(PKG / "cases" / "manifest.csv", manifest_rows)
    bad = [r for r in manifest_rows if r.get("mutation_verified") is False]
    print(f"registry rows={len(registry_rows)} splits={len(split_rows)} cases={len(manifest_rows)} failed_verifications={len(bad)}")
    for r in bad:
        print("  FAILED", r["case_id"], r["verification_errors"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
