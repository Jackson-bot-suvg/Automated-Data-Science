"""Same-prefix fork: snapshot a run right before a gate node, resume branches from it.

Snapshot (prefix run, raised from the gate node's observer before the gate runs):
  <fork_dir>/state.json            GraphState.model_dump(mode="json") as seen by the gate
  <fork_dir>/workspace/            copy of the run workspace (artifacts/, hooks/)
  <fork_dir>/prefix_events.jsonl   events of the prefix run up to the fork point
  <fork_dir>/rng.pkl               python + numpy global RNG state at the fork point
  <fork_dir>/fork_meta.json        state_hash, candidate_hash, contract_digest,
                                   visible_evidence_hash, remaining_budget, rng hash
All hashes use run-dir-normalised paths so branch copies hash identically.
"""

from __future__ import annotations

import pickle
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from lane_a.common import read_json, sha256_bytes, sha256_json, write_json
from lane_a.context import ctx
from lane_a.state_views import candidate_hash, contract_digest, state_hash

RUN_DIR_TOKEN = "<RUN_DIR>"


def _rng_blob() -> bytes:
    return pickle.dumps({"python": random.getstate(), "numpy": np.random.get_state()})


def rng_hash() -> str:
    return sha256_bytes(_rng_blob())


def remaining_budget(state) -> dict[str, Any]:
    c = ctx()
    return {
        "retry_count_by_phase": dict(state.review_state.retry_count_by_phase or {}),
        "llm_calls_used": c.llm_calls,
        "llm_call_budget": c.llm_budget,
        "max_attempts_per_phase": 3,
    }


def write_snapshot(state, *, node_name: str, phase: str) -> Path:
    c = ctx()
    fork_dir = Path(c.fork["fork_dir"])
    fork_dir.mkdir(parents=True, exist_ok=True)
    (fork_dir / "state.json").write_text(state.model_dump_json(), encoding="utf-8")
    ws = fork_dir / "workspace"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for sub in ("artifacts", "hooks"):
        src = c.run_dir / sub
        if src.exists():
            shutil.copytree(src, ws / sub)
    (fork_dir / "rng.pkl").write_bytes(_rng_blob())
    meta = {
        "prefix_run_id": c.run_id,
        "prefix_run_dir": str(c.run_dir),
        "gate_node": node_name,
        "gate_phase": phase,
        "occurrence": int(c.fork.get("occurrence", 1)),
        "state_hash": state_hash(state, {str(c.run_dir): RUN_DIR_TOKEN}),
        "candidate_hash": candidate_hash(state, phase, {str(c.run_dir): RUN_DIR_TOKEN}),
        "contract_digest": contract_digest(state),
        "visible_evidence_hash": c.fork.get("visible_evidence_hash", ""),
        "remaining_budget": remaining_budget(state),
        "rng_state_sha256": rng_hash(),
        "injections_applied": list(c.injections_applied),
        "fit_seconds_prefix": round(c.fit_seconds_total, 3),
        "llm_calls_prefix": c.llm_calls,
    }
    write_json(fork_dir / "fork_meta.json", meta)
    c.logger.emit(
        "fork_snapshot",
        actor="lane_a_fork",
        phase=phase,
        state_hash=meta["state_hash"],
        candidate_hash=meta["candidate_hash"],
        contract_digest=meta["contract_digest"],
        payload={k: meta[k] for k in ("gate_node", "gate_phase", "occurrence", "visible_evidence_hash", "remaining_budget", "rng_state_sha256")},
    )
    # events are copied after the fork_snapshot event has been written
    shutil.copy2(c.logger.path, fork_dir / "prefix_events.jsonl")
    return fork_dir


def restore_workspace(fork_dir: Path, run_dir: Path) -> None:
    ws = Path(fork_dir) / "workspace"
    for sub in ("artifacts", "hooks"):
        src = ws / sub
        if src.exists():
            shutil.copytree(src, run_dir / sub, dirs_exist_ok=True)


def load_state_json(fork_dir: Path, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = read_json(Path(fork_dir) / "fork_meta.json")
    text = (Path(fork_dir) / "state.json").read_text(encoding="utf-8")
    text = text.replace(meta["prefix_run_dir"], str(run_dir))
    import json

    return json.loads(text), meta


def restore_rng(fork_dir: Path) -> None:
    blob = pickle.loads((Path(fork_dir) / "rng.pkl").read_bytes())
    random.setstate(blob["python"])
    np.random.set_state(blob["numpy"])


__all__ = ["RUN_DIR_TOKEN", "load_state_json", "restore_rng", "restore_workspace", "rng_hash", "write_snapshot", "sha256_json"]
