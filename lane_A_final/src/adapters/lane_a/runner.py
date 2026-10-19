"""Single-run entry point (one process per run).

    python -P -m lane_a.runner --spec <run_spec.json>

The process chdirs into the run workspace, installs (in this order) tool-error
capture, the LLM observer, node/action observers with any declared injection
inside them, and the method's gate policy; then runs the F0 LangGraph workflow
(technical-only, as in the manuscript's Table 2 protocol).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata as md
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from lane_a import ADAPTER_VERSION
from lane_a.common import EventLogger, read_json, sha256_file, write_json
from lane_a.context import ForkPointReached, LLMBudgetExceeded, RunContext, set_ctx

ENV_REPORT_KEYS = [
    "LLM_PROVIDERS",
    "LLM_DEEPSEEK_BASE_URL",
    "LLM_DEEPSEEK_TYPE",
    "AGENT_READER_MODEL",
    "AGENT_PLANNER_MODEL",
    "AGENT_BG_REVIEWER_MODEL",
    "AGENT_REVIEWER_MODEL",
    "AGENT_SUMMARIZER_MODEL",
    "AGENT_READER_TEMPERATURE",
    "AGENT_PLANNER_TEMPERATURE",
    "AGENT_BG_REVIEWER_TEMPERATURE",
    "AGENT_REVIEWER_TEMPERATURE",
    "AGENT_READER_MAX_TOKENS",
    "AGENT_PLANNER_MAX_TOKENS",
    "AGENT_BG_REVIEWER_MAX_TOKENS",
    "AGENT_REVIEWER_MAX_TOKENS",
    "LANGSMITH_TRACING",
]


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("langgraph", "langchain-core", "scikit-learn", "pandas", "numpy", "xgboost", "lightgbm", "pydantic", "openai"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def _build_graph(spec: dict[str, Any]):
    import app.graph.workflow as wf

    technical_only = bool(spec.get("technical_only", True))
    fork = spec.get("fork") or {}
    if fork.get("mode") == "continue":
        from langgraph.graph import START, StateGraph

        gate_node = fork["gate_node"]
        orig_add_edge = StateGraph.add_edge

        def add_edge(self, start_key, end_key):
            if start_key == START and end_key == "reader":
                return orig_add_edge(self, START, gate_node)
            return orig_add_edge(self, start_key, end_key)

        StateGraph.add_edge = add_edge
        try:
            return wf.build_workflow(stop_after_technical_summary=technical_only)
        finally:
            StateGraph.add_edge = orig_add_edge
    return wf.build_workflow(stop_after_technical_summary=technical_only)


def _install_naive_gate_removal() -> None:
    """Reconstruction of the manuscript's 'No reviewer gates' ablation (Table 3).

    Gate nodes are replaced by pass-through nodes that return no state update; all
    edges and routing functions are unchanged.  This is a hypothesis about how the
    old ablation was built (its code is not in the base commit), used only in A0.
    """

    import app.graph.workflow as wf

    async def passthrough_reviewer(state):
        return {}

    async def passthrough_bg_reviewer(state):
        return {}

    wf.reviewer_node = passthrough_reviewer
    wf.bg_reviewer_node = passthrough_bg_reviewer


def _state_from_values(values: Any):
    from app.schemas.state_schema import GraphState

    if values is None:
        return None
    if isinstance(values, GraphState):
        return values
    if isinstance(values, dict):
        try:
            return GraphState(**values)
        except Exception:  # noqa: BLE001
            return None
    return None


async def _drive(graph, state, *, recursion_limit: int):
    holder: dict[str, Any] = {"last": None}
    async for values in graph.astream(state, stream_mode="values", config={"recursion_limit": recursion_limit}):
        holder["last"] = values
    return holder["last"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    run_dir = Path(spec["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    if (run_dir / "run_result.json").exists():
        print(f"run {spec['run_id']} already completed; refusing to overwrite", file=sys.stderr)
        return 3
    seed = int(spec.get("seed_block") or 0)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    hooks_dir = run_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    budget = dict(spec.get("budget") or {})
    logger = EventLogger(run_dir / "events.jsonl", spec["run_id"])
    ctx = RunContext(
        run_id=spec["run_id"],
        run_dir=run_dir,
        hooks_dir=hooks_dir,
        logger=logger,
        method_id=spec["method_id"],
        case_id=spec.get("case_id", ""),
        template_id=spec.get("template_id", "C00"),
        injection=dict(spec.get("injection") or {}),
        fork=dict(spec.get("fork") or {}),
        seed_block=seed,
        path_replacements={str(run_dir): "<RUN_DIR>"},
        llm_budget=int(budget.get("llm_calls", 60)),
    )
    set_ctx(ctx)

    from lane_a import hooks, injections, methods

    env_report = {k: os.environ.get(k, "") for k in ENV_REPORT_KEYS}
    logger.emit(
        "run_start",
        actor="lane_a_runner",
        payload={
            "spec": {k: v for k, v in spec.items() if k not in {"secrets"}},
            "adapter_version": ADAPTER_VERSION,
            "versions": _versions(),
            "env": env_report,
            "pid": os.getpid(),
            "cwd": str(run_dir),
            "app_module_path": __import__("app").__path__[0],
        },
    )

    observers = bool(spec.get("observers", True))
    llm_mode = spec.get("llm_mode", "live")
    method = spec["method_id"]
    injection = dict(spec.get("injection") or {})
    if observers:
        hooks.install_tool_error_capture()
        hooks.install_llm_observer(mode=llm_mode)
        if method == "M0_NAIVE_GATE_REMOVAL":
            _install_naive_gate_removal()
        hooks.install_node_observers(inner_wrappers=injections.node_inner_wrappers(injection))
        hooks.install_action_observers(inner_wrappers=injections.action_inner_wrappers(injection))
        methods.install_method(method)
    else:
        # unobserved baseline for the hook-equivalence smoke: no lane_a wrappers at all,
        # except the identical LLM-off switch needed for a deterministic replay.
        if llm_mode == "off":
            from app.llm.service import LLMService

            async def _off(self, model, messages, **kwargs):
                raise RuntimeError("LLM disabled (deterministic replay mode)")

            LLMService.chat = _off
        if method != "F0_ORIGINAL" or injection:
            raise SystemExit("unobserved mode is only defined for plain F0 without injection")

    fork = ctx.fork
    t_start = time.monotonic()
    outcome = "completed_graph"
    error_info: dict[str, Any] = {}
    state = None
    final_values = None
    try:
        if fork.get("mode") == "continue":
            from app.schemas.state_schema import GraphState
            from lane_a import fork as fork_mod
            from lane_a.state_views import candidate_hash, contract_digest, state_hash

            fork_dir = Path(fork["fork_dir"])
            fork_mod.restore_workspace(fork_dir, run_dir)
            shutil.copy2(fork_dir / "prefix_events.jsonl", run_dir / "prefix_events.jsonl")
            data, meta = fork_mod.load_state_json(fork_dir, run_dir)
            state = GraphState(**data)
            fork_mod.restore_rng(fork_dir)
            ctx.injections_applied = list(meta.get("injections_applied") or [])
            ctx.llm_calls = int(meta.get("llm_calls_prefix") or 0)
            resumed_state_hash = state_hash(state, ctx.path_replacements)
            resumed_candidate = candidate_hash(state, meta["gate_phase"], ctx.path_replacements)
            logger.emit(
                "fork_resume",
                actor="lane_a_fork",
                phase=meta["gate_phase"],
                state_hash=resumed_state_hash,
                candidate_hash=resumed_candidate,
                contract_digest=contract_digest(state),
                payload={
                    "prefix_run_id": meta["prefix_run_id"],
                    "snapshot_state_hash": meta["state_hash"],
                    "snapshot_candidate_hash": meta["candidate_hash"],
                    "state_hash_match": resumed_state_hash == meta["state_hash"],
                    "candidate_hash_match": resumed_candidate == meta["candidate_hash"],
                    "rng_state_sha256": fork_mod.rng_hash(),
                    "rng_match": fork_mod.rng_hash() == meta["rng_state_sha256"],
                    "remaining_budget": meta["remaining_budget"],
                    "visible_evidence_hash": meta.get("visible_evidence_hash"),
                    "gate_node": meta["gate_node"],
                },
            )
        else:
            from app.intake import build_initial_state

            task = spec["task"]
            state = build_initial_state(
                task=task["task"],
                dataset=task["dataset"],
                target=task.get("target", ""),
                metric_hint=task.get("metric_hint", ""),
                workspace_root=".",
                thread_id=spec["run_id"],
            )
        graph = _build_graph(spec)
        final_values = asyncio.run(
            asyncio.wait_for(
                _drive(graph, state, recursion_limit=int(budget.get("recursion_limit", 80))),
                timeout=float(budget.get("wall_seconds", 1800)),
            )
        )
    except ForkPointReached as exc:
        outcome = "forked"
        error_info = {"message": str(exc)}
    except LLMBudgetExceeded as exc:
        outcome = "budget_llm_calls"
        error_info = {"message": str(exc)}
    except (asyncio.TimeoutError, TimeoutError) as exc:
        outcome = "budget_wall_clock"
        error_info = {"message": repr(exc)}
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        outcome = "recursion_limit" if name == "GraphRecursionError" else "exception"
        tb = traceback.format_exc()
        tb_path = run_dir / "errors" / "run_exception.txt"
        tb_path.parent.mkdir(parents=True, exist_ok=True)
        tb_path.write_text(tb, encoding="utf-8")
        error_info = {"type": name, "message": str(exc)[:4000], "traceback_path": str(tb_path)}
        logger.emit("exception", actor="lane_a_runner", error_subtype=name, payload=error_info)

    wall = round(time.monotonic() - t_start, 3)
    final_state = _state_from_values(final_values)
    result: dict[str, Any] = {
        "run_id": spec["run_id"],
        "outcome": outcome,
        "error": error_info,
        "wall_seconds": wall,
        "llm_calls": ctx.llm_calls,
        "llm_errors": ctx.llm_errors,
        "fit_seconds_total": round(ctx.fit_seconds_total, 3),
        "injections_applied": ctx.injections_applied,
    }
    if final_state is not None:
        (run_dir / "final_state.json").write_text(final_state.model_dump_json(indent=2), encoding="utf-8")
        rs = final_state.review_state
        pred_rel = str(final_state.artifacts.get("prediction_file") or "")
        pred_path = (run_dir / pred_rel) if pred_rel and not Path(pred_rel).is_absolute() else Path(pred_rel) if pred_rel else None
        completed = final_state.current_phase in {"technical_summary"} or bool(final_state.report_state.technical_report_path)
        result.update(
            {
                "final_phase": final_state.current_phase,
                "latest_review_status": rs.latest_review_status,
                "pending_review_phase": rs.pending_review_phase,
                "retry_count_by_phase": dict(rs.retry_count_by_phase),
                "fatal_errors": list(rs.fatal_errors),
                "technical_report_path": final_state.report_state.technical_report_path,
                "workflow_completed": bool(completed and rs.latest_review_status == "approved"),
                "workflow_aborted": rs.latest_review_status == "fatal",
                "prediction_path": str(pred_path) if pred_path else "",
                "prediction_exists": bool(pred_path and pred_path.exists()),
                "prediction_sha256": sha256_file(pred_path) if pred_path and pred_path.exists() else "",
                "model_state_prediction_path": final_state.model_state.prediction_path,
                "review_history": [
                    {k: rec.get(k) for k in ("phase", "status", "attempt", "blocking_failures")}
                    for rec in rs.review_history
                ],
            }
        )
    logger.emit("run_end", actor="lane_a_runner", payload=result)
    write_json(run_dir / "run_result.json", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
