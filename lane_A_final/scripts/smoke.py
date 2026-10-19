#!/usr/bin/env python3
"""Minimal checks with automatic gate output (no API key needed).

1. read-only hook / method-wrapper behaviour equivalence (A0-EQUIV, LLM-off replay):
   the unobserved F0 run vs every observed variant on a clean task must agree on node
   sequence, final contract, review statuses and the exact submission bytes
   -> results/A0/version_diff.csv
2. mutation verification status of every case in cases/manifest.csv
3. oracle unit tests (src/oracle/test_oracle.py)
Writes results/A0/smoke_checks.json and exits non-zero if any check fails.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PKG, WORK  # noqa: E402

OUT = PKG / "results" / "A0"


def _final(run_dir: Path) -> dict:
    fs = json.loads((run_dir / "final_state.json").read_text())
    res = json.loads((run_dir / "run_result.json").read_text())
    ts = fs["target_spec"]
    contract = {k: ts[k] for k in ("target_columns", "task_family", "target_mode", "prediction_kind")}
    contract["submission"] = ts["submission"]["submission_columns"]
    contract["metric"] = ts["evaluation"]["primary_metric"]
    statuses = [(r["phase"], r["status"], r["attempt"]) for r in fs["review_state"]["review_history"]]
    mr_path = None
    for p in run_dir.rglob("model_results.json"):
        mr_path = p
    stable_mr = None
    if mr_path:
        mr = json.loads(mr_path.read_text())
        stable_mr = hashlib.sha256(json.dumps({k: mr.get(k) for k in ("candidate_models", "best_model_name", "best_params", "_meta")}, sort_keys=True).encode()).hexdigest()
    return {
        "phase_history": fs["phase_history"],
        "contract": contract,
        "review_statuses": statuses,
        "submission_sha256": res.get("prediction_sha256"),
        "model_results_stable_sha256": stable_mr,
        "outcome": res.get("outcome"),
    }


def equivalence() -> tuple[bool, list[dict]]:
    runs = sorted(p for p in (WORK / "runs").glob("A0E__*") if (p / "run_result.json").exists())
    ref_dir = next((p for p in runs if "F0_UNOBSERVED" in p.name), None)
    rows = []
    if ref_dir is None:
        return False, [{"error": "unobserved reference run missing"}]
    ref = _final(ref_dir)
    ok_all = True
    for p in runs:
        cur = _final(p)
        diffs = [k for k in ref if ref[k] != cur[k]]
        ok = not diffs
        ok_all &= ok
        rows.append({
            "run_id": p.name,
            "reference_run": ref_dir.name,
            "identical": ok,
            "differing_fields": "|".join(diffs),
            "submission_sha256": cur["submission_sha256"],
            "phase_history_len": len(cur["phase_history"]),
            "review_statuses": json.dumps(cur["review_statuses"]),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "version_diff.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return ok_all, rows


def mutations() -> tuple[bool, int]:
    with open(PKG / "cases" / "manifest.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("applicability", "") not in {"N/A"}]
    bad = [r for r in rows if r.get("mutation_verified") == "False"]
    return not bad, len(rows)


def oracle_tests() -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(PKG / "src" / "oracle" / "test_oracle.py"), "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=PKG, env={"PYTHONPATH": f"{PKG / 'src'}", "PATH": "/usr/bin:/bin"})
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-1500:]


def main() -> int:
    eq_ok, eq_rows = equivalence()
    mut_ok, n_cases = mutations()
    or_ok, or_out = oracle_tests()
    result = {
        "hook_and_method_equivalence_llm_off": {"pass": eq_ok, "runs_compared": len(eq_rows)},
        "mutations_verified": {"pass": mut_ok, "cases": n_cases},
        "oracle_unit_tests": {"pass": or_ok, "tail": or_out},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["pass"] for k, v in result.items()}, indent=2))
    return 0 if (eq_ok and mut_ok and or_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
