"""Read-only observers.

Every wrapper calls the original function with the original arguments and returns
the original return value unchanged; the only side effects are appends to
``events.jsonl`` / ``llm_calls.jsonl`` and copies of predict/serialize frames in
the lane-private ``hooks/`` directory (never read by the agent).  Behaviour
equivalence is checked in ``scripts/smoke.py`` (deterministic LLM-off replay).
"""

from __future__ import annotations

import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from lane_a.common import (
    canonical_json,
    ordered_values_digest,
    pairing_digest,
    series_fingerprint,
    sha256_file,
    sha256_json,
    sha256_text,
    utc_now,
)
from lane_a.context import ForkPointReached, LLMBudgetExceeded, ctx
from lane_a.state_views import candidate_hash, contract_digest, contract_view, state_hash

GRAPH_NODE_FUNCS = {
    "reader": "reader_node",
    "background_reviewer": "bg_reviewer_node",
    "planner": "planner_node",
    "preliminary_eda": "prelim_eda_node",
    "data_cleaning": "data_cleaning_node",
    "indepth_eda": "indepth_eda_node",
    "feature_engineering": "feature_engineering_node",
    "model_building": "modeling_node",
    "reviewer": "reviewer_node",
    "summarizer": "summarizer_node",
}

TOOL_ERROR_RE = re.compile(
    r"^\[(?P<phase>[a-z_]+)\]\s+(?P<kind>runtime_error|tool_execution_failed|artifact_missing|missing_context|target_column_lost|schema_error):\s*(?P<msg>.*)$",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Tool error capture (phase error envelopes are reported through logging.error)
# ---------------------------------------------------------------------------


class _ToolErrorHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        m = TOOL_ERROR_RE.match(msg or "")
        if not m:
            return
        try:
            ctx().logger.emit(
                "tool_error",
                phase=m.group("phase"),
                actor=record.name,
                error_subtype=m.group("kind"),
                payload={"message": m.group("msg")[:2000]},
            )
        except Exception:  # noqa: BLE001
            pass


def install_tool_error_capture() -> None:
    handler = _ToolErrorHandler(level=logging.ERROR)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(min(logging.getLogger().level or logging.WARNING, logging.WARNING))


# ---------------------------------------------------------------------------
# Graph node observers (+ fork interception for prefix runs)
# ---------------------------------------------------------------------------


def _update_summary(updates: Any) -> dict[str, Any]:
    if not isinstance(updates, dict):
        return {"type": type(updates).__name__}
    out: dict[str, Any] = {"keys": sorted(updates.keys())}
    rs = updates.get("review_state")
    if isinstance(rs, dict):
        out["latest_review_status"] = rs.get("latest_review_status")
        out["retry_count_by_phase"] = rs.get("retry_count_by_phase")
    rt = updates.get("runtime_state")
    if isinstance(rt, dict) and rt.get("interrupted"):
        out["interrupted"] = True
    if "current_phase" in updates:
        out["current_phase"] = updates.get("current_phase")
    return out


def _plan_for_phase(updates: dict[str, Any], phase: str) -> dict[str, Any]:
    plan = ((updates.get("execution_plan") or {}).get("success_criteria") or {}).get(phase) or {}
    return plan if isinstance(plan, dict) else {}


def wrap_node(node_name: str, fn):
    async def wrapped(state):
        c = ctx()
        phase = state.current_phase
        _maybe_fork(node_name, state)
        c.logger.emit(
            "node_start",
            actor=node_name,
            phase=phase,
            state_hash=state_hash(state, c.path_replacements),
            contract_digest=contract_digest(state),
        )
        t0 = time.perf_counter()
        try:
            updates = await fn(state)
        except (ForkPointReached, LLMBudgetExceeded):
            raise
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            tb_path = c.run_dir / "errors" / f"{node_name}_{int(time.time()*1000)}.txt"
            tb_path.parent.mkdir(parents=True, exist_ok=True)
            tb_path.write_text(tb, encoding="utf-8")
            c.logger.emit(
                "exception",
                actor=node_name,
                phase=phase,
                error_subtype=type(exc).__name__,
                payload={"message": str(exc)[:2000], "traceback_path": str(tb_path)},
            )
            raise
        elapsed = round(time.perf_counter() - t0, 4)
        new_phase = updates.get("current_phase", phase) if isinstance(updates, dict) else phase
        c.current_phase = new_phase or phase
        extra: dict[str, Any] = {"elapsed_s": elapsed, "update": _update_summary(updates)}
        fields: dict[str, Any] = {}
        if node_name == "reader" and isinstance(updates, dict) and isinstance(updates.get("target_spec"), dict):
            ts = updates["target_spec"]
            extra["contract_read_candidate"] = {
                "status": ts.get("status"),
                "target_columns": ts.get("target_columns"),
                "task_family": ts.get("task_family"),
                "target_mode": ts.get("target_mode"),
                "prediction_kind": ts.get("prediction_kind"),
                "primary_metric": (ts.get("evaluation") or {}).get("primary_metric"),
                "id_columns": (ts.get("submission") or {}).get("id_columns"),
                "output_columns": (ts.get("submission") or {}).get("output_columns"),
                "submission_columns": (ts.get("submission") or {}).get("submission_columns"),
                "evidence": ts.get("evidence"),
            }
            fields["contract_digest"] = sha256_json(extra["contract_read_candidate"])
        if node_name == "planner" and isinstance(updates, dict):
            plan = _plan_for_phase(updates, new_phase)
            agent_plan = plan.get("agent_plan") if isinstance(plan, dict) else None
            extra["planned_phase"] = new_phase
            extra["plan"] = agent_plan
            fields["candidate_hash"] = sha256_json(agent_plan or {})
        c.logger.emit("node_end", actor=node_name, phase=new_phase, payload=extra, **fields)
        return updates

    wrapped.__name__ = getattr(fn, "__name__", node_name)
    wrapped.__lane_a_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def _maybe_fork(node_name: str, state) -> None:
    c = ctx()
    spec = c.fork or {}
    if spec.get("mode") != "prefix":
        return
    if node_name != spec.get("gate_node"):
        return
    if (state.current_phase or "reader") != spec.get("gate_phase"):
        return
    key = f"fork:{node_name}:{spec.get('gate_phase')}"
    c.gate_counter[key] = c.gate_counter.get(key, 0) + 1
    if c.gate_counter[key] != int(spec.get("occurrence", 1)):
        return
    from lane_a.fork import write_snapshot

    write_snapshot(state, node_name=node_name, phase=spec.get("gate_phase"))
    raise ForkPointReached(f"fork point reached before {node_name}@{spec.get('gate_phase')}")


def install_node_observers(*, inner_wrappers: dict[str, Any] | None = None) -> None:
    """Patch node callables in app.graph.workflow before build_workflow() is called.

    ``inner_wrappers`` lets injections sit *inside* the observer, so observers log
    exactly what the graph receives.
    """

    import app.graph.workflow as wf

    inner_wrappers = inner_wrappers or {}
    for node_name, attr in GRAPH_NODE_FUNCS.items():
        fn = getattr(wf, attr)
        if getattr(fn, "__lane_a_wrapped__", False):
            continue
        if node_name in inner_wrappers:
            fn = inner_wrappers[node_name](fn)
        setattr(wf, attr, wrap_node(node_name, fn))


# ---------------------------------------------------------------------------
# LLM call observer (also enforces the frozen per-run call budget)
# ---------------------------------------------------------------------------


def install_llm_observer(*, mode: str = "live") -> None:
    from app.llm.service import LLMService

    orig_chat = LLMService.chat
    raw_path_holder: dict[str, Path] = {}

    async def chat(self, model, messages, **kwargs):
        c = ctx()
        raw_path = raw_path_holder.setdefault("p", c.run_dir / "llm_calls.jsonl")
        module = str(kwargs.get("module_name") or "unknown")
        prompt_text = canonical_json([{"role": m.role, "content": m.content} for m in messages])
        call_no = c.llm_calls + 1
        params = {k: kwargs.get(k) for k in ("temperature", "max_tokens", "top_p", "stream") if k in kwargs}
        if mode == "off":
            c.logger.emit(
                "llm_call",
                actor=module,
                phase=c.current_phase,
                error_subtype="llm_disabled_replay",
                payload={"call_no": call_no, "model": model, "params": params, "prompt_sha256": sha256_text(prompt_text)},
            )
            raise RuntimeError("LLM disabled (deterministic replay mode)")
        if c.llm_calls >= c.llm_budget:
            c.logger.emit(
                "budget_exhausted",
                actor=module,
                phase=c.current_phase,
                error_subtype="llm_call_budget",
                payload={"llm_calls": c.llm_calls, "budget": c.llm_budget},
            )
            raise LLMBudgetExceeded(f"LLM call budget {c.llm_budget} exhausted")
        c.llm_calls += 1
        t0 = time.perf_counter()
        started = utc_now()
        try:
            result = await orig_chat(self, model, messages, **kwargs)
        except Exception as exc:  # noqa: BLE001
            c.llm_errors += 1
            c.logger.emit(
                "llm_call",
                actor=module,
                phase=c.current_phase,
                error_subtype=type(exc).__name__,
                payload={
                    "call_no": call_no,
                    "model": model,
                    "params": params,
                    "prompt_sha256": sha256_text(prompt_text),
                    "latency_s": round(time.perf_counter() - t0, 3),
                    "error": str(exc)[:1000],
                },
            )
            with open(raw_path, "a", encoding="utf-8") as fh:
                fh.write(canonical_json({"call_no": call_no, "module": module, "model": model, "started": started, "params": params, "messages": prompt_text, "error": str(exc)[:4000]}) + "\n")
            raise
        latency = round(time.perf_counter() - t0, 3)
        content = getattr(result, "content", None)
        usage = getattr(result, "usage", None)
        usage_d = None
        if usage is not None:
            usage_d = {"input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None)}
        returned_model = getattr(result, "model", None)
        c.logger.emit(
            "llm_call",
            actor=module,
            phase=c.current_phase,
            payload={
                "call_no": call_no,
                "model": model,
                "returned_model": returned_model,
                "params": params,
                "prompt_sha256": sha256_text(prompt_text),
                "response_sha256": sha256_text(content or ""),
                "response_chars": len(content or ""),
                "usage": usage_d,
                "latency_s": latency,
            },
        )
        with open(raw_path, "a", encoding="utf-8") as fh:
            fh.write(
                canonical_json(
                    {
                        "call_no": call_no,
                        "module": module,
                        "model": model,
                        "returned_model": returned_model,
                        "started": started,
                        "params": params,
                        "messages": prompt_text,
                        "response": content,
                        "usage": usage_d,
                    }
                )
                + "\n"
            )
        return result

    LLMService.chat = chat


# ---------------------------------------------------------------------------
# Actual fit / predict / serialization / write observers (model_building namespace)
# ---------------------------------------------------------------------------


def install_action_observers(*, inner_wrappers: dict[str, Any] | None = None) -> None:
    import app.agents.model_building as mb

    inner_wrappers = inner_wrappers or {}
    orig_fit = mb.fit_and_select_model
    orig_predict = mb.predict_for_submission
    orig_build = inner_wrappers.get("build_submission", lambda f: f)(mb.build_submission)
    orig_save = mb.save_model_artifacts

    def fit_and_select_model(X_train, y_train, preprocessor, candidates, **kwargs):
        c = ctx()
        y_cols = [str(y_train.name)] if isinstance(y_train, pd.Series) else [str(col) for col in y_train.columns]
        fp = series_fingerprint(y_train)
        actual_hash = sha256_json({"target_columns": y_cols, "fingerprint": fp["digest"]})
        c.logger.emit(
            "fit_start",
            actor="model_building",
            phase="model_building",
            actual_fit_target_hash=actual_hash,
            payload={
                "y_columns": y_cols,
                "y_fingerprint": fp,
                "y_ordered_digest": ordered_values_digest(y_train),
                "n_rows": int(len(X_train)),
                "x_columns": [str(col) for col in X_train.columns],
                "x_columns_sha256": sha256_json([str(col) for col in X_train.columns]),
                "task_family": kwargs.get("task_family"),
                "target_mode": kwargs.get("target_mode"),
                "primary_metric": kwargs.get("primary_metric"),
                "multi_target_subtype": kwargs.get("multi_target_subtype"),
                "candidates": sorted(list(candidates.keys())) if isinstance(candidates, dict) else None,
            },
        )
        t0 = time.perf_counter()
        result = orig_fit(X_train, y_train, preprocessor, candidates, **kwargs)
        dt = round(time.perf_counter() - t0, 3)
        c.fit_seconds_total += dt
        best_pipeline, best_name, all_results = result
        summary = {}
        for name, info in (all_results or {}).items():
            if isinstance(info, dict):
                summary[name] = {k: v for k, v in info.items() if isinstance(v, (int, float, str)) and k != "best_params"}
        c.logger.emit(
            "fit_end",
            actor="model_building",
            phase="model_building",
            actual_fit_target_hash=actual_hash,
            payload={"fit_seconds": dt, "best_model": best_name, "results": summary},
        )
        return result

    def predict_for_submission(pipeline, X_test, **kwargs):
        c = ctx()
        preds = orig_predict(pipeline, X_test, **kwargs)
        classes = getattr(pipeline, "classes_", None)
        if classes is None and hasattr(pipeline, "named_steps"):
            est = pipeline.named_steps.get("classifier")
            classes = getattr(est, "classes_", None)
        try:
            classes_list = [str(x) for x in list(classes)] if classes is not None else None
        except TypeError:
            classes_list = None
        out_path = c.hooks_dir / "predict_output.csv"
        preds.to_csv(out_path, index=False)
        c.last_predict = {"columns": [str(col) for col in preds.columns], "path": str(out_path)}
        c.logger.emit(
            "predict",
            actor="model_building",
            phase="model_building",
            payload={
                "output_columns_arg": kwargs.get("output_columns"),
                "prediction_kind": kwargs.get("prediction_kind"),
                "task_family": kwargs.get("task_family"),
                "pred_columns": [str(col) for col in preds.columns],
                "n_rows": int(len(preds)),
                "pred_ordered_digest": ordered_values_digest(preds),
                "classes_order": classes_list,
                "hook_file": str(out_path),
                "hook_file_sha256": sha256_file(out_path),
            },
        )
        return preds

    def build_submission(test_df, predictions, **kwargs):
        c = ctx()
        frame = orig_build(test_df, predictions, **kwargs)
        id_cols = list(kwargs.get("id_columns") or [])
        out_cols = list(kwargs.get("output_columns") or list(predictions.columns))
        sub_cols = list(kwargs.get("submission_columns") or [])
        present_ids = [col for col in id_cols if col in test_df.columns]
        synthesized = [col for col in id_cols if col not in test_df.columns]
        id_frame_path = c.hooks_dir / "reference_ids.csv"
        if present_ids:
            test_df[present_ids].reset_index(drop=True).to_csv(id_frame_path, index=False)
        frame_path = c.hooks_dir / "submission_frame.csv"
        frame.to_csv(frame_path, index=False)
        non_id = [col for col in frame.columns if col not in id_cols]
        provenance = {}
        preds_reset = predictions.reset_index(drop=True)
        for col in frame.columns:
            if col in present_ids:
                provenance[col] = "test_id"
            elif col in synthesized:
                provenance[col] = "synthesized_range_id"
            elif col in preds_reset.columns:
                same = frame[col].reset_index(drop=True).astype(str).equals(preds_reset[col].astype(str))
                provenance[col] = "prediction" if same else "prediction_reordered_or_altered"
            else:
                provenance[col] = "nan_filled"
        c.logger.emit(
            "serialize",
            actor="model_building",
            phase="model_building",
            artifact_mapping_digest=pairing_digest(frame, present_ids, non_id),
            payload={
                "id_columns": id_cols,
                "output_columns": out_cols,
                "submission_columns": sub_cols,
                "frame_columns": [str(col) for col in frame.columns],
                "column_provenance": provenance,
                "n_rows": int(len(frame)),
                "hook_frame_file": str(frame_path),
                "hook_frame_sha256": sha256_file(frame_path),
                "reference_ids_file": str(id_frame_path) if present_ids else "",
            },
        )
        return frame

    def save_model_artifacts(best_pipeline, model_results_data, submission_df, **kwargs):
        c = ctx()
        result = orig_save(best_pipeline, model_results_data, submission_df, **kwargs)
        pred_path = (result or {}).get("prediction_file") or ""
        payload: dict[str, Any] = {"paths": result}
        if pred_path and Path(pred_path).exists():
            try:
                df = pd.read_csv(pred_path)
                payload.update({"prediction_sha256": sha256_file(pred_path), "shape": list(df.shape), "columns": list(df.columns)})
            except Exception as exc:  # noqa: BLE001
                payload["read_error"] = str(exc)
        c.logger.emit("artifact_write", actor="model_building", phase="model_building", payload=payload)
        return result

    mb.fit_and_select_model = fit_and_select_model
    mb.predict_for_submission = predict_for_submission
    mb.build_submission = build_submission
    mb.save_model_artifacts = save_model_artifacts


def emit_contract_snapshot(event_type: str, state) -> None:
    c = ctx()
    c.logger.emit(
        event_type,
        phase=state.current_phase,
        actor="lane_a",
        state_hash=state_hash(state, c.path_replacements),
        contract_digest=contract_digest(state),
        payload={"contract": contract_view(state)},
    )


__all__ = [
    "install_action_observers",
    "install_llm_observer",
    "install_node_observers",
    "install_tool_error_capture",
    "emit_contract_snapshot",
    "candidate_hash",
]
