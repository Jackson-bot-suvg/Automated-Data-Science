"""Helpers for persisting a compact per-run manifest artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_manifest(
    *,
    workspace_root: str,
    manifest_path: str,
    run_id: str,
    thread_id: str,
    current_phase: str,
    phase_history: list[str],
    task_config: dict[str, Any],
    target_spec: dict[str, Any],
    artifacts: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
    data_state: dict[str, Any] | None = None,
    feature_state: dict[str, Any] | None = None,
    model_state: dict[str, Any] | None = None,
    review_state: dict[str, Any] | None = None,
    report_state: dict[str, Any] | None = None,
) -> str:
    """Write a compact run manifest for artifact and state inspection."""

    target = Path(manifest_path)
    if not target.is_absolute():
        target = Path(workspace_root or ".").resolve() / target
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": str(run_id or "").strip(),
        "thread_id": str(thread_id or "").strip(),
        "current_phase": str(current_phase or "").strip(),
        "phase_history": list(phase_history or []),
        "task": {
            "task_id": str(task_config.get("task_id") or "").strip(),
            "raw_task": str(task_config.get("raw_task") or "").strip(),
            "dataset_root": str(task_config.get("dataset_root") or "").strip(),
            "target_hint": str(task_config.get("target_hint") or "").strip(),
        },
        "target_spec": {
            "status": target_spec.get("status"),
            "task_family": target_spec.get("task_family"),
            "target_mode": target_spec.get("target_mode"),
            "prediction_kind": target_spec.get("prediction_kind"),
            "target_columns": list(target_spec.get("target_columns") or []),
            "submission": target_spec.get("submission") or {},
            "evaluation": target_spec.get("evaluation") or {},
        },
        "artifacts": {
            key: value
            for key, value in dict(artifacts or {}).items()
            if key != "figures"
        }
        | {"figures": list(dict(artifacts or {}).get("figures") or [])},
    }

    if data_state:
        manifest["data_state"] = {
            "current_train_path": data_state.get("current_train_path", ""),
            "current_test_path": data_state.get("current_test_path", ""),
            "train_shape": data_state.get("train_shape"),
            "test_shape": data_state.get("test_shape"),
            "data_quality_flags": list(data_state.get("data_quality_flags") or []),
        }
    if runtime_state:
        manifest["runtime_state"] = {
            "thread_id": runtime_state.get("thread_id", ""),
            "run_id": runtime_state.get("run_id", ""),
            "artifact_root": runtime_state.get("artifact_root", ""),
            "phase_runtime_sec": runtime_state.get("phase_runtime_sec") or {},
            "total_runtime_sec": runtime_state.get("total_runtime_sec", 0.0),
            "last_reviewed_phase": runtime_state.get("last_reviewed_phase", ""),
        }
    if feature_state:
        manifest["feature_state"] = {
            "engineered_train_path": feature_state.get("engineered_train_path", ""),
            "engineered_test_path": feature_state.get("engineered_test_path", ""),
            "selected_features": list(feature_state.get("selected_features") or []),
            "dropped_features": list(feature_state.get("dropped_features") or []),
            "leakage_checks": list(feature_state.get("leakage_checks") or []),
        }
    if model_state:
        manifest["model_state"] = {
            "candidate_models": list(model_state.get("candidate_models") or []),
            "best_model_name": model_state.get("best_model_name", ""),
            "best_params": model_state.get("best_params") or {},
            "validation_metrics": model_state.get("validation_metrics") or {},
            "baseline_metrics": model_state.get("baseline_metrics") or {},
        }
    if review_state:
        review_history = list(review_state.get("review_history") or [])
        review_history_summary: list[dict[str, Any]] = []
        for record in review_history:
            if not isinstance(record, dict):
                continue
            review_history_summary.append(
                {
                    "phase": record.get("phase"),
                    "status": record.get("status"),
                    "attempt": record.get("attempt"),
                    "blocking_failures": list(record.get("blocking_failures") or []),
                    "warnings": list(record.get("warnings") or []),
                }
            )
        manifest["review_state"] = {
            "latest_review_status": review_state.get("latest_review_status"),
            "last_review_comment": review_state.get("last_review_comment", ""),
            "pending_review_phase": review_state.get("pending_review_phase", ""),
            "retry_count_by_phase": review_state.get("retry_count_by_phase") or {},
            "fatal_errors": list(review_state.get("fatal_errors") or []),
            "review_history_summary": review_history_summary,
        }
    if report_state:
        manifest["report_state"] = {
            "technical_report_path": report_state.get("technical_report_path", ""),
            "business_report_path": report_state.get("business_report_path", ""),
            "limitations": list(report_state.get("limitations") or []),
            "recommended_actions": list(report_state.get("recommended_actions") or []),
        }

    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return str(target)
