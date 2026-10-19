#!/usr/bin/env python3
"""Lane-A run orchestrator.

    python scripts/run.py --dry-run                     # print the full matrix, N/A cells and a cost estimate
    python scripts/run.py --pilot --stage A0 A1 A2 A3   # run DEV cells (resumable; never overwrites)
    python scripts/run.py --resume ...                  # same as --pilot; completed run_ids are skipped
    python scripts/run.py --confirm                     # CONF cells; refuses to start while G0 is not PASS

One OS process per run (python -P -m lane_a.runner --spec ...), cwd = its own run
workspace, PYTHONPATH = <base clone>:<adapters>.  Within each seed block the run order
is shuffled with a recorded seed.  A2/A3 continuation branches are launched only after
their prefix has written a fork snapshot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ADAPTERS, BASE, BASE_COMMIT, DATA_ROOT, DEV_SEED_BLOCKS, CONF_SEED_BLOCKS, PKG, WORK  # noqa: E402
from specs import (  # noqa: E402
    A2_BRANCHES,
    A2_CONTEXTS,
    A3_CONTEXTS,
    DEV_TASKS,
    G_NA_CONTEXTS,
    PILOT_TASKS,
    PROBE_PARAMS,
)

LOGS = PKG / "logs"
PLAN_LEDGER = LOGS / "plan_ledger.csv"
RUN_INDEX = LOGS / "run_index.csv"
FROZEN_CONFIG = PKG / "protocol" / "frozen_config.yaml"
BUDGET = {"llm_calls": 60, "wall_seconds": 1800, "recursion_limit": 80}
HARD_KILL_SECONDS = 2100
ORDER_SEED = 20261019
MODEL_ID = "deepseek/deepseek-flash"
HOST_ID = "host-macA-14c-24g"
A1_TEMPLATES = ["C00", "B01", "F03", "F01", "F05", "F10"]
A0_SMOKE_TASKS = ["DEV-BIN-TITANIC", "DEV-MC-GHOULS"]
A0_EQUIV_METHODS = ["F0_ORIGINAL", "M0_NO_SEMANTIC_ENFORCEMENT", "M2_INTAKE_ONLY_STRONG", "M3_SHADOW", "M4_END_ONLY", "M5_BLOCK_ONLY", "G_GENERIC_RETRY"]

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def run_env(seed: int) -> dict[str, str]:
    env_file = Path(os.environ.get("LANE_A_ENV_FILE", PKG.parents[1] / ".env"))
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "SHELL"}}
    env.update(load_env_file(env_file))
    env.update(
        {
            "PYTHONPATH": f"{BASE}{os.pathsep}{ADAPTERS}",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "MPLBACKEND": "Agg",
            "PYTHONHASHSEED": str(seed),
            "OMP_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _pkg(task_id: str, template: str) -> str:
    if template in {"F05", "F10"}:
        template = "C00"
    return str(WORK / "packages" / task_id / template)


def _injection(task_id: str, template: str, seed: int) -> dict:
    if template == "F05":
        return {"template_id": "F05", "wrong_target_columns": [PROBE_PARAMS[task_id]["drift_target"]]}
    if template == "F10":
        return {"template_id": "F10", "permutation_seed": seed}
    return {}


def _task_block(task_id: str, template: str) -> dict:
    spec = DEV_TASKS[task_id]
    hint = PROBE_PARAMS[task_id]["wrong_hint"] if template == "F03" else spec["hint_target"]
    return {"task": spec["q"], "dataset": _pkg(task_id, template), "target": hint, "metric_hint": spec["metric_hint"]}


def _visible_hash(case_id: str) -> str:
    manifest = PKG / "cases" / "manifest.csv"
    with open(manifest, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["case_id"] == case_id:
                return row.get("visible_evidence_hash", "")
    return ""


def cell(**kw) -> dict:
    base = {
        "lane_id": "A",
        "host_id": HOST_ID,
        "model_id": MODEL_ID,
        "freeze_sha": "UNSIGNED(PRE-FREEZE)",
        "planned_status": "planned",
        "applicability": "applicable",
        "n_a_reason": "",
        "source_group_id": "",
        "case_id": "",
        "template_id": "",
        "condition": "",
        "pair_id": "",
        "depends_on": "",
    }
    base.update(kw)
    if base["task_id"] in DEV_TASKS:
        base["source_group_id"] = f"SRC-{DEV_TASKS[base['task_id']]['data_dir']}"
    return base


def build_plan() -> list[dict]:
    cells: list[dict] = []
    # ---------------- A0 ----------------
    for task_id, spec in DEV_TASKS.items():
        for rep in (1, 2, 3):
            rid = f"A0C__{task_id}__C00__F0_ORIGINAL__sb1101__r{rep}"
            cells.append(
                cell(
                    plan_cell_id=rid, stage="A0-CLEAN", task_id=task_id, case_id=f"{task_id}:ORIG", template_id="C00",
                    condition="clean_original_package", method_id="F0_ORIGINAL", seed_block=1101, repeat=rep,
                    spec=dict(mode="full", task={"task": spec["q"], "dataset": str(DATA_ROOT / spec["data_dir"]), "target": spec["hint_target"], "metric_hint": spec["metric_hint"]}),
                )
            )
    for task_id in A0_SMOKE_TASKS:
        spec = DEV_TASKS[task_id]
        for method in ("M0_NO_SEMANTIC_ENFORCEMENT", "M3_SHADOW", "M0_NAIVE_GATE_REMOVAL"):
            rid = f"A0S__{task_id}__C00__{method}__sb1101__r1"
            cells.append(
                cell(
                    plan_cell_id=rid, stage="A0-SMOKE", task_id=task_id, case_id=f"{task_id}:ORIG", template_id="C00",
                    condition="clean_original_package", method_id=method, seed_block=1101, repeat=1,
                    spec=dict(mode="full", task={"task": spec["q"], "dataset": str(DATA_ROOT / spec["data_dir"]), "target": spec["hint_target"], "metric_hint": spec["metric_hint"]}),
                )
            )
    eq_task = "DEV-BIN-TITANIC"
    spec = DEV_TASKS[eq_task]
    eq_block = {"task": spec["q"], "dataset": str(DATA_ROOT / spec["data_dir"]), "target": spec["hint_target"], "metric_hint": spec["metric_hint"]}
    cells.append(
        cell(plan_cell_id="A0E__DEV-BIN-TITANIC__C00__F0_UNOBSERVED__llmoff", stage="A0-EQUIV", task_id=eq_task, case_id=f"{eq_task}:ORIG",
             template_id="C00", condition="clean_original_package", method_id="F0_ORIGINAL", seed_block=1101, repeat=1,
             spec=dict(mode="full", task=eq_block, llm_mode="off", observers=False))
    )
    for method in A0_EQUIV_METHODS:
        cells.append(
            cell(plan_cell_id=f"A0E__DEV-BIN-TITANIC__C00__{method}__llmoff", stage="A0-EQUIV", task_id=eq_task, case_id=f"{eq_task}:ORIG",
                 template_id="C00", condition="clean_original_package", method_id=method, seed_block=1101, repeat=1,
                 spec=dict(mode="full", task=eq_block, llm_mode="off", observers=True))
        )
    for template in ("C00", "B01", "F03", "F01", "F05"):
        task_id = "DEV-ML-STEEL"
        rid = f"A0P__{task_id}__{template}__F0_ORIGINAL__sb1101__r1"
        cells.append(
            cell(plan_cell_id=rid, stage="A0-PROBE", task_id=task_id, case_id=f"{task_id}:{template}", template_id=template,
                 condition=template, method_id="F0_ORIGINAL", seed_block=1101, repeat=1,
                 spec=dict(mode="full", task=_task_block(task_id, template), injection=_injection(task_id, template, 1101)))
        )
    # ---------------- A1 ----------------
    for family, task_id in PILOT_TASKS.items():
        for template in A1_TEMPLATES:
            for method in ("F0_ORIGINAL", "M3_SHADOW"):
                if task_id is None:
                    cells.append(cell(plan_cell_id=f"A1__ML-none__{template}__{method}", stage="A1", task_id="DEV-ML-(none)", case_id=f"DEV-ML-(none):{template}",
                                      template_id=template, method_id=method, seed_block=1101, repeat=1, applicability="N/A", planned_status="N/A",
                                      n_a_reason="no genuine multilabel DEV task (Steel co-occurrence 0.11%); requires CONF-ML", spec=None))
                    continue
                rid = f"A1__{task_id}__{template}__{method}__sb1101"
                cells.append(cell(plan_cell_id=rid, stage="A1", task_id=task_id, case_id=f"{task_id}:{template}", template_id=template,
                                  condition=template, method_id=method, seed_block=1101, repeat=1, pair_id=f"A1PAIR__{task_id}__{template}",
                                  spec=dict(mode="full", task=_task_block(task_id, template), injection=_injection(task_id, template, 1101))))
    # ---------------- A2 / A3 DEV pilot ----------------
    for family, task_id in PILOT_TASKS.items():
        for ctx_name, ctx in A2_CONTEXTS.items():
            for sb in DEV_SEED_BLOCKS:
                template = ctx["template"]
                prefix_id = f"A2P__{task_id or 'ML-none'}__{ctx_name}__sb{sb}"
                if task_id is None:
                    for method in A2_BRANCHES:
                        cells.append(cell(plan_cell_id=f"A2__ML-none__{ctx_name}__sb{sb}__{method}", stage="A2-PILOT", task_id="DEV-ML-(none)", template_id=template,
                                          method_id=method, seed_block=sb, repeat=1, applicability="N/A", planned_status="N/A",
                                          n_a_reason="no genuine multilabel DEV task", spec=None))
                    continue
                fork_dir = WORK / "forks" / prefix_id
                cells.append(cell(plan_cell_id=prefix_id, stage="A2-PREFIX", task_id=task_id, case_id=f"{task_id}:{template}", template_id=template,
                                  condition=ctx_name, method_id="PREFIX(F0 up to gate)", seed_block=sb, repeat=1, pair_id=prefix_id,
                                  spec=dict(mode="prefix", task=_task_block(task_id, template), injection=_injection(task_id, template, sb),
                                            fork={"mode": "prefix", "gate_node": ctx["gate_node"], "gate_phase": ctx["gate_phase"], "occurrence": 1,
                                                  "fork_dir": str(fork_dir), "visible_evidence_hash": _visible_hash(f"{task_id}:{template}")})))
                branches = list(A2_BRANCHES) + (["M2_INTAKE_ONLY_STRONG"] if ctx_name in A3_CONTEXTS else [])
                for method in branches:
                    stage = "A3-PILOT" if method == "M2_INTAKE_ONLY_STRONG" else "A2-PILOT"
                    rid = f"A2__{task_id}__{ctx_name}__sb{sb}__{method}"
                    if method == "G_GENERIC_RETRY" and ctx_name in G_NA_CONTEXTS:
                        cells.append(cell(plan_cell_id=rid, stage=stage, task_id=task_id, case_id=f"{task_id}:{template}", template_id=template, condition=ctx_name,
                                          method_id=method, seed_block=sb, repeat=1, pair_id=prefix_id, applicability="N/A", planned_status="N/A",
                                          n_a_reason=G_NA_CONTEXTS[ctx_name], spec=None))
                        continue
                    cells.append(cell(plan_cell_id=rid, stage=stage, task_id=task_id, case_id=f"{task_id}:{template}", template_id=template, condition=ctx_name,
                                      method_id=method, seed_block=sb, repeat=1, pair_id=prefix_id, depends_on=prefix_id,
                                      spec=dict(mode="continue", task=_task_block(task_id, template), injection=_injection(task_id, template, sb),
                                                fork={"mode": "continue", "fork_dir": str(fork_dir), "gate_node": ctx["gate_node"], "gate_phase": ctx["gate_phase"]})))
    # ---------------- formal CONF (blocked by G0) ----------------
    for fam in ("BIN", "MC", "ML", "REG"):
        for k in (1, 2, 3):
            task_id = f"CONF-{fam}-0{k}"
            for ctx_name in A2_CONTEXTS:
                for sb in CONF_SEED_BLOCKS:
                    prefix_id = f"A2P__{task_id}__{ctx_name}__sb{sb}"
                    cells.append(cell(plan_cell_id=prefix_id, stage="A2-CONF-PREFIX", task_id=task_id, template_id=A2_CONTEXTS[ctx_name]["template"], condition=ctx_name,
                                      method_id="PREFIX", seed_block=sb, repeat=1, planned_status="not_started", applicability="blocked_G0",
                                      n_a_reason="G0 FAIL: FREEZE_SIGNOFF.yaml unsigned; CONF task identities not frozen", spec=None))
                    for method in list(A2_BRANCHES) + (["M2_INTAKE_ONLY_STRONG"] if ctx_name in A3_CONTEXTS else []):
                        cells.append(cell(plan_cell_id=f"A2__{task_id}__{ctx_name}__sb{sb}__{method}", stage="A3-CONF" if method.startswith("M2") else "A2-CONF",
                                          task_id=task_id, template_id=A2_CONTEXTS[ctx_name]["template"], condition=ctx_name, method_id=method, seed_block=sb,
                                          repeat=1, pair_id=prefix_id, planned_status="not_started", applicability="blocked_G0",
                                          n_a_reason="G0 FAIL: FREEZE_SIGNOFF.yaml unsigned; CONF task identities not frozen"
                                          + ("; G pre-registered N/A for this context" if method == "G_GENERIC_RETRY" and ctx_name in G_NA_CONTEXTS else ""),
                                          spec=None))
    return cells


LEDGER_FIELDS = [
    "plan_cell_id", "lane_id", "stage", "task_id", "source_group_id", "condition", "template_id", "case_id", "method_id",
    "host_id", "model_id", "seed_block", "repeat", "pair_id", "depends_on", "planned_status", "applicability", "n_a_reason", "freeze_sha",
    "run_status", "run_id", "order_index",
]


def write_ledger(cells: list[dict]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    status = _run_statuses()
    with open(PLAN_LEDGER, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            row = dict(c)
            st = status.get(c["plan_cell_id"])
            row["run_status"] = st or ("not_started" if c["planned_status"] not in {"N/A"} else "N/A")
            row["run_id"] = c["plan_cell_id"] if st else ""
            w.writerow(row)


def _run_statuses() -> dict[str, str]:
    out = {}
    runs = WORK / "runs"
    if not runs.exists():
        return out
    for d in runs.iterdir():
        res = d / "run_result.json"
        if res.exists():
            try:
                out[d.name] = "completed:" + json.loads(res.read_text()).get("outcome", "?")
            except Exception:  # noqa: BLE001
                out[d.name] = "completed:unreadable"
        elif (d / "events.jsonl").exists():
            out[d.name] = "started_incomplete"
    return out


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def make_spec(c: dict) -> dict:
    s = dict(c["spec"])
    s.update(
        {
            "run_id": c["plan_cell_id"],
            "run_dir": str(WORK / "runs" / c["plan_cell_id"]),
            "lane": "A",
            "stage": c["stage"],
            "plan_cell_id": c["plan_cell_id"],
            "pair_id": c.get("pair_id", ""),
            "task_id": c["task_id"],
            "case_id": c.get("case_id", ""),
            "template_id": c.get("template_id", ""),
            "condition": c.get("condition", ""),
            "method_id": c["method_id"] if not str(c["method_id"]).startswith("PREFIX") else "F0_ORIGINAL",
            "seed_block": c["seed_block"],
            "budget": dict(BUDGET),
            "technical_only": True,
            "base_commit": BASE_COMMIT,
            "model_id": MODEL_ID,
            "host_id": HOST_ID,
        }
    )
    s.setdefault("llm_mode", "live")
    s.setdefault("observers", True)
    s.setdefault("injection", {})
    return s


def launch(c: dict) -> dict:
    rid = c["plan_cell_id"]
    run_dir = WORK / "runs" / rid
    if (run_dir / "run_result.json").exists():
        return {"run_id": rid, "status": "skipped_completed"}
    if run_dir.exists() and any(run_dir.iterdir()):
        # never overwrite a started run: keep it as an infra record and stop here
        return {"run_id": rid, "status": "skipped_started_incomplete"}
    spec = make_spec(c)
    spec_path = WORK / "specs" / f"{rid}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True))
    run_dir.mkdir(parents=True, exist_ok=True)
    py = os.environ.get("LANE_A_PYTHON", str(PKG.parents[1] / ".venv" / "bin" / "python"))
    cmd = [py, "-P", "-m", "lane_a.runner", "--spec", str(spec_path)]
    t0 = time.time()
    with open(run_dir / "stdout.log", "w", encoding="utf-8") as out:
        try:
            proc = subprocess.run(cmd, cwd=run_dir, env=run_env(int(c["seed_block"])), stdout=out, stderr=subprocess.STDOUT, timeout=HARD_KILL_SECONDS)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    wall = round(time.time() - t0, 1)
    if not (run_dir / "run_result.json").exists():
        # process died without writing its result (kill / crash): record an infra stub, keep logs
        stub = {"run_id": rid, "outcome": "infra_process_failure", "returncode": rc, "wall_seconds": wall}
        (run_dir / "run_result.json").write_text(json.dumps(stub, indent=2))
    res = json.loads((run_dir / "run_result.json").read_text())
    row = {
        "run_id": rid, "plan_cell_id": rid, "stage": c["stage"], "task_id": c["task_id"], "method_id": spec["method_id"],
        "template_id": c.get("template_id", ""), "seed_block": c["seed_block"], "returncode": rc, "outcome": res.get("outcome"),
        "wall_seconds": wall, "run_dir": str(run_dir), "spec_sha256": __import__("hashlib").sha256(spec_path.read_bytes()).hexdigest(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with _lock:
        new = not RUN_INDEX.exists()
        with open(RUN_INDEX, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
    print(f"[{time.strftime('%H:%M:%S')}] {rid} -> {res.get('outcome')} ({wall}s)", flush=True)
    return {"run_id": rid, "status": res.get("outcome")}


def ordered(cells: list[dict]) -> list[dict]:
    """Shuffle within seed blocks with a recorded seed; blocks keep their numeric order."""

    rng = random.Random(ORDER_SEED)
    by_block: dict[int, list[dict]] = {}
    for c in cells:
        by_block.setdefault(int(c["seed_block"]), []).append(c)
    out = []
    for sb in sorted(by_block):
        block = sorted(by_block[sb], key=lambda x: x["plan_cell_id"])
        rng.shuffle(block)
        out.extend(block)
    for i, c in enumerate(out):
        c["order_index"] = i
    return out


def run_cells(cells: list[dict], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(launch, c) for c in cells]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:  # noqa: BLE001
                print("launcher error:", exc, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--confirm", action="store_true")
    ap.add_argument("--stage", nargs="*", default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to plan_cell_ids containing any of these substrings")
    args = ap.parse_args()

    cells = build_plan()
    write_ledger(cells)
    runnable = [c for c in cells if c["planned_status"] == "planned" and c["spec"] is not None]

    def in_stage(c: dict) -> bool:
        return any(c["stage"].startswith(s) for s in args.stage)

    if args.dry_run:
        from collections import Counter

        cnt = Counter((c["stage"], c["planned_status"]) for c in cells)
        for (stage, st), n in sorted(cnt.items()):
            print(f"{stage:16s} {st:12s} {n:5d}")
        live = [c for c in runnable if c["spec"].get("llm_mode", "live") == "live"]
        print(f"runnable DEV cells: {len(runnable)} (LLM-live {len(live)})")
        print(f"cost estimate: ~{len(live)*36_000/1e6:.1f}M input + {len(live)*12_000/1e6:.1f}M output tokens on {MODEL_ID} (A0 pilot measurement ~36k/12k per technical-only run; continuations are cheaper)")
        print(f"plan ledger written: {PLAN_LEDGER}")
        return 0
    if args.confirm:
        print("REFUSED: G0 is FAIL (FREEZE_SIGNOFF.yaml unsigned; CONF tasks not frozen). Only DEV pilot cells may run.")
        return 2

    selected = [c for c in runnable if in_stage(c)]
    if args.only:
        selected = [c for c in selected if any(s in c["plan_cell_id"] for s in args.only)]
    prefixes = ordered([c for c in selected if c["stage"] == "A2-PREFIX"])
    singles = ordered([c for c in selected if not c.get("depends_on") and c["stage"] != "A2-PREFIX"])
    conts = [c for c in selected if c.get("depends_on")]
    print(f"selected: singles={len(singles)} prefixes={len(prefixes)} continuations={len(conts)} workers={args.workers}", flush=True)
    run_cells(singles + prefixes, args.workers)
    ready = []
    for c in conts:
        fork_meta = Path(c["spec"]["fork"]["fork_dir"]) / "fork_meta.json"
        if fork_meta.exists():
            ready.append(c)
        else:
            print(f"prefix not available for {c['plan_cell_id']} (prefix outcome recorded separately)", flush=True)
    run_cells(ordered(ready), args.workers)
    write_ledger(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
