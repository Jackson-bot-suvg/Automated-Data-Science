"""Hashable views of GraphState used for state/contract/candidate digests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lane_a.common import normalize_paths, sha256_file, sha256_json


def contract_view(state: Any) -> dict[str, Any]:
    ts = state.target_spec
    return {
        "status": ts.status,
        "target_columns": list(ts.target_columns),
        "task_family": ts.task_family,
        "target_mode": ts.target_mode,
        "prediction_kind": ts.prediction_kind,
        "primary_metric": ts.evaluation.primary_metric,
        "candidate_metrics": list(ts.evaluation.candidate_metrics),
        "id_columns": list(ts.submission.id_columns),
        "output_columns": list(ts.submission.output_columns),
        "submission_columns": list(ts.submission.submission_columns),
    }


def contract_digest(state: Any) -> str:
    return sha256_json(contract_view(state))


_VOLATILE_RUNTIME = {"phase_runtime_sec", "total_runtime_sec", "token_usage", "llm_calls", "cost_estimate"}


def semantic_state_view(state: Any, replacements: dict[str, str] | None = None) -> dict[str, Any]:
    """Full state minus volatile timing/usage fields, with run-dir paths normalised."""

    data = state.model_dump(mode="json")
    runtime = dict(data.get("runtime_state") or {})
    for key in _VOLATILE_RUNTIME:
        runtime.pop(key, None)
    data["runtime_state"] = runtime
    data.pop("messages", None)
    review = dict(data.get("review_state") or {})
    # LLM-phrased comments are not decision inputs; keep the deterministic record parts only.
    review.pop("last_review_comment", None)
    history = []
    for record in review.get("review_history") or []:
        rec = dict(record)
        rec.pop("comment", None)
        history.append(rec)
    review["review_history"] = history
    data["review_state"] = review
    if replacements:
        data = normalize_paths(data, replacements)
    return data


def state_hash(state: Any, replacements: dict[str, str] | None = None) -> str:
    return sha256_json(semantic_state_view(state, replacements))


_PHASE_ARTIFACT_KEYS = {
    "reader": [],
    "preliminary_eda": ["preliminary_eda_report"],
    "data_cleaning": ["cleaned_train_data", "cleaned_test_data", "cleaning_report"],
    "indepth_eda": ["indepth_eda_report"],
    "feature_engineering": ["feature_summary", "engineered_train_data", "engineered_test_data"],
    "model_building": ["model_results", "prediction_file"],
}


def _resolve(workspace_root: str, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else Path(workspace_root or ".") / p


def candidate_view(state: Any, phase: str, replacements: dict[str, str] | None = None) -> dict[str, Any]:
    """What the gate is about to judge: phase outputs (state slice + artifact file hashes)."""

    data = state.model_dump(mode="json")
    view: dict[str, Any] = {"phase": phase, "contract": contract_view(state)}
    if phase == "reader":
        view["problem_understanding"] = data.get("problem_understanding")
        view["task_config"] = data.get("task_config")
    elif phase in {"data_cleaning", "preliminary_eda", "indepth_eda"}:
        view["data_state"] = data.get("data_state")
    elif phase == "feature_engineering":
        view["feature_state"] = data.get("feature_state")
    elif phase == "model_building":
        view["model_state"] = data.get("model_state")
    files = {}
    for key in _PHASE_ARTIFACT_KEYS.get(phase, []):
        raw = str((state.artifacts or {}).get(key) or "")
        if not raw:
            continue
        resolved = _resolve(state.runtime_state.workspace_root, raw)
        if resolved.exists() and resolved.is_file():
            if key == "model_results":
                # timing fields inside model_results are volatile; hash the stable subset
                import json

                try:
                    payload = json.loads(resolved.read_text())
                    stable = {
                        "candidate_models": payload.get("candidate_models"),
                        "best_model_name": payload.get("best_model_name"),
                        "best_params": payload.get("best_params"),
                        "_meta": payload.get("_meta"),
                    }
                    files[key] = sha256_json(stable)
                except Exception:
                    files[key] = sha256_file(resolved)
            else:
                files[key] = sha256_file(resolved)
    view["artifact_sha256"] = files
    if replacements:
        view = normalize_paths(view, replacements)
    return view


def candidate_hash(state: Any, phase: str, replacements: dict[str, str] | None = None) -> str:
    return sha256_json(candidate_view(state, phase, replacements))
