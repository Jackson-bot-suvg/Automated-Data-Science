"""Deterministic reviewer checks for phase artifacts and state evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas.reviewer_schema import ReviewCheckResult
from app.schemas.state_schema import GraphState
from app.utils.target_scope import canonical_target_scope, normalize_target_columns


@dataclass(frozen=True)
class _ResolvedTargetContract:
    target_columns: list[str]
    id_columns: list[str]
    output_columns: list[str]
    submission_columns: list[str]
    target_mode: str
    task_family: str
    prediction_kind: str
    used_legacy_fallback: bool = False


def resolve_artifact_path(workspace_root: str, artifact_path: str) -> Path:
    """Resolve an artifact path against the workflow workspace root."""

    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return Path(workspace_root or ".").joinpath(path).resolve()


def artifact_exists(workspace_root: str, artifact_path: str) -> bool:
    """Return whether an artifact path is present on disk."""

    if not artifact_path:
        return False
    return resolve_artifact_path(workspace_root, artifact_path).exists()


def csv_shape(workspace_root: str, artifact_path: str) -> tuple[int, int] | None:
    """Return CSV shape, or None when the path is missing/unreadable."""

    if not artifact_path:
        return None
    try:
        return tuple(pd.read_csv(resolve_artifact_path(workspace_root, artifact_path)).shape)
    except Exception:
        return None


def csv_columns(workspace_root: str, artifact_path: str) -> list[str]:
    """Return CSV columns, or an empty list when the path is missing/unreadable."""

    if not artifact_path:
        return []
    try:
        return pd.read_csv(resolve_artifact_path(workspace_root, artifact_path), nrows=0).columns.tolist()
    except Exception:
        return []


def review_preliminary_eda(state: GraphState) -> ReviewCheckResult:
    """Check that Preliminary EDA produced the context required by later phases."""

    result = ReviewCheckResult(phase="preliminary_eda")
    workspace_root = state.runtime_state.workspace_root

    _require_artifact(state, result, "preliminary_eda_report")
    _mark_state_fields(
        result,
        "data_state.schema_summary",
        "data_state.eda_findings",
        "artifacts.figures",
    )

    if not state.data_state.schema_summary:
        result.blocking_failures.append("data_state.schema_summary is empty after Preliminary EDA")
    if not state.data_state.eda_findings:
        result.blocking_failures.append("data_state.eda_findings is empty after Preliminary EDA")
    else:
        _warn_missing_profile_keys(result, state.data_state.eda_findings)

    if not state.artifacts.get("figures"):
        result.warnings.append("artifacts.figures is empty; EDA still passes if structured findings exist")

    _record_csv_details(result, "current_train_path", workspace_root, state.data_state.current_train_path)
    return result


def review_data_cleaning(state: GraphState) -> ReviewCheckResult:
    """Check cleaned data artifacts and the state handoff into feature engineering."""

    result = ReviewCheckResult(phase="data_cleaning")
    workspace_root = state.runtime_state.workspace_root
    has_external_test = _has_external_test(state)
    target_contract = _resolve_target_contract(state)

    _require_artifact(state, result, "cleaned_train_data")
    _require_phase_record(state, result, bucket="phase_plan_records", phase="data_cleaning")
    if has_external_test:
        _require_artifact(state, result, "cleaned_test_data")
    else:
        result.warnings.append("No external test data; cleaned_test_data is optional")
    _require_artifact(state, result, "cleaning_report")
    _mark_state_fields(
        result,
        "data_state.current_train_path",
        "data_state.current_test_path",
        "data_state.cleaning_summary",
        "data_state.data_quality_flags",
        "target_spec.status",
        "target_spec.target_columns",
        "target_spec.target_mode",
        "target_spec.task_family",
        "target_spec.submission.id_columns",
        "target_spec.submission.output_columns",
        "target_spec.submission.submission_columns",
    )
    _record_target_contract(result, target_contract)

    cleaned_train = str(state.artifacts.get("cleaned_train_data", ""))
    cleaned_test = str(state.artifacts.get("cleaned_test_data", ""))
    plan_record = _read_phase_record(
        state,
        bucket="phase_plan_records",
        phase="data_cleaning",
    )
    _check_data_cleaning_plan_health(result, plan_record)

    if cleaned_train and not _same_path(workspace_root, cleaned_train, state.data_state.current_train_path):
        result.blocking_failures.append(
            "data_state.current_train_path does not point to artifacts.cleaned_train_data"
        )
    if has_external_test and cleaned_test and not _same_path(workspace_root, cleaned_test, state.data_state.current_test_path):
        result.blocking_failures.append(
            "data_state.current_test_path does not point to artifacts.cleaned_test_data"
        )

    _require_readable_non_empty_csv(result, workspace_root, cleaned_train, "cleaned_train_data")
    if has_external_test:
        _require_readable_non_empty_csv(result, workspace_root, cleaned_test, "cleaned_test_data")

    cleaned_train_columns = csv_columns(workspace_root, cleaned_train)
    cleaned_test_columns = csv_columns(workspace_root, cleaned_test)

    if not target_contract.target_columns:
        result.blocking_failures.append(
            "target contract unresolved: reviewer needs target_spec.target_columns or a compatibility mirror"
        )
    else:
        missing_targets = [
            column
            for column in target_contract.target_columns
            if column not in cleaned_train_columns
        ]
        if missing_targets:
            result.blocking_failures.append(
                "cleaned training data is missing target columns "
                + ", ".join(missing_targets)
            )
        leaked_targets = [
            column
            for column in target_contract.target_columns
            if column in cleaned_test_columns
        ]
        if leaked_targets:
            result.blocking_failures.append(
                "cleaned test data still contains target columns "
                + ", ".join(leaked_targets)
            )
    if not state.data_state.cleaning_summary:
        result.warnings.append("data_state.cleaning_summary is empty; downstream traceability is weak")
    if not state.data_state.data_quality_flags:
        result.warnings.append("data_state.data_quality_flags is empty; no quality notes were recorded")

    return result

def review_indepth_eda(state: GraphState) -> ReviewCheckResult:
    """Check that In-Depth EDA produced the context required by later phases."""

    result = ReviewCheckResult(phase="indepth_eda")
    workspace_root = state.runtime_state.workspace_root

    _require_artifact(state, result, "indepth_eda_report")
    _require_phase_record(state, result, bucket="phase_plan_records", phase="indepth_eda")
    _mark_state_fields(
        result,
        "data_state.indepth_eda_summary",
        "artifacts.indepth_eda_report",
        "artifacts.figures",
    )

    if not state.data_state.indepth_eda_summary:
        result.blocking_failures.append("data_state.indepth_eda_summary is empty after In-Depth EDA")
    else:
        _warn_missing_profile_keys(result, state.data_state.eda_findings)

    if not state.artifacts.get("figures"):
        result.warnings.append("artifacts.figures is empty; EDA still passes if structured findings exist")

    _record_csv_details(result, "current_train_path", workspace_root, state.data_state.current_train_path)
    return result

def review_feature_engineering(state: GraphState) -> ReviewCheckResult:
    """Check engineered feature artifacts and train/test feature alignment."""

    result = ReviewCheckResult(phase="feature_engineering")
    workspace_root = state.runtime_state.workspace_root
    has_external_test = _has_external_test(state)
    target_contract = _resolve_target_contract(state)

    _require_artifact(state, result, "feature_summary")
    _require_phase_record(state, result, bucket="phase_plan_records", phase="feature_engineering")
    _mark_state_fields(
        result,
        "feature_state.engineered_train_path",
        "feature_state.engineered_test_path",
        "feature_state.selected_features",
        "feature_state.feature_transform_summary",
        "feature_state.leakage_checks",
        "target_spec.status",
        "target_spec.target_columns",
        "target_spec.target_mode",
        "target_spec.task_family",
        "target_spec.submission.id_columns",
    )
    _record_target_contract(result, target_contract)

    train_path = state.feature_state.engineered_train_path
    test_path = state.feature_state.engineered_test_path
    _require_file(result, workspace_root, train_path, "feature_state.engineered_train_path")
    if has_external_test:
        _require_file(result, workspace_root, test_path, "feature_state.engineered_test_path")
    else:
        result.warnings.append("No external test data; engineered_test_path is optional")
    _require_readable_non_empty_csv(result, workspace_root, train_path, "engineered_train_path")
    if has_external_test:
        _require_readable_non_empty_csv(result, workspace_root, test_path, "engineered_test_path")

    if not state.feature_state.selected_features:
        result.blocking_failures.append("feature_state.selected_features is empty")
    if not state.feature_state.feature_transform_summary:
        result.warnings.append("feature_state.feature_transform_summary is empty")
    if not state.feature_state.leakage_checks:
        result.warnings.append("feature_state.leakage_checks is empty; reviewer cannot confirm leakage checks")

    train_columns_all = csv_columns(workspace_root, train_path)
    test_columns_all = csv_columns(workspace_root, test_path)

    if not target_contract.target_columns:
        result.blocking_failures.append(
            "target contract unresolved: reviewer needs target_spec.target_columns or a compatibility mirror"
        )
        return result

    missing_targets = [
        column
        for column in target_contract.target_columns
        if column not in train_columns_all
    ]
    if missing_targets:
        result.blocking_failures.append(
            "engineered training data is missing target columns "
            + ", ".join(missing_targets)
        )

    leaked_targets = [
        column
        for column in target_contract.target_columns
        if column in test_columns_all
    ]
    if leaked_targets:
        result.blocking_failures.append(
            "engineered test data still contains target columns "
            + ", ".join(leaked_targets)
        )

    selected_features = list(state.feature_state.selected_features or [])
    target_features = [
        column for column in selected_features if column in target_contract.target_columns
    ]
    if target_features:
        result.blocking_failures.append(
            "feature_state.selected_features contains target columns "
            + ", ".join(target_features)
        )

    id_features = [
        column for column in selected_features if column in target_contract.id_columns
    ]
    if id_features:
        result.warnings.append(
            "feature_state.selected_features includes submission id columns "
            + ", ".join(id_features)
        )

    excluded_columns = set(target_contract.target_columns) | set(target_contract.id_columns)
    train_columns = _feature_columns(train_columns_all, excluded_columns)
    test_columns = _feature_columns(test_columns_all, excluded_columns)
    missing_selected = [column for column in selected_features if column not in train_columns_all]
    if missing_selected:
        result.blocking_failures.append(
            "feature_state.selected_features references missing engineered columns "
            + ", ".join(missing_selected)
        )

    if has_external_test and train_columns and test_columns and set(train_columns) != set(test_columns):
        missing_in_test = sorted(set(train_columns) - set(test_columns))
        extra_in_test = sorted(set(test_columns) - set(train_columns))
        result.blocking_failures.append(
            f"engineered train/test feature columns mismatch; missing_in_test={missing_in_test}, "
            f"extra_in_test={extra_in_test}"
        )

    return result


def review_model_building(state: GraphState) -> ReviewCheckResult:
    """Check model artifacts, prediction output, and basic submission shape."""

    result = ReviewCheckResult(phase="model_building")
    workspace_root = state.runtime_state.workspace_root
    has_external_test = _has_external_test(state)
    target_contract = _resolve_target_contract(state)

    _require_artifact(state, result, "model_results")
    _require_phase_record(state, result, bucket="phase_plan_records", phase="model_building")
    _require_artifact(state, result, "best_model")
    if has_external_test:
        _require_artifact(state, result, "prediction_file")
    else:
        result.warnings.append("No external test data; prediction_file is optional")
    _mark_state_fields(
        result,
        "model_state.candidate_models",
        "model_state.best_model_name",
        "model_state.validation_metrics",
        "model_state.cv_results",
        "model_state.prediction_path",
        "model_state.baseline_metrics",
        "target_spec.status",
        "target_spec.target_columns",
        "target_spec.target_mode",
        "target_spec.task_family",
        "target_spec.submission.id_columns",
        "target_spec.submission.output_columns",
        "target_spec.submission.submission_columns",
    )
    _record_target_contract(result, target_contract)

    prediction_artifact = str(state.artifacts.get("prediction_file", ""))
    if has_external_test and prediction_artifact and not _same_path(workspace_root, prediction_artifact, state.model_state.prediction_path):
        result.blocking_failures.append("model_state.prediction_path does not point to artifacts.prediction_file")

    if not state.model_state.candidate_models:
        result.blocking_failures.append("model_state.candidate_models is empty")
    if not state.model_state.best_model_name:
        result.blocking_failures.append("model_state.best_model_name is empty")
    elif (
        state.model_state.candidate_models
        and state.model_state.best_model_name not in state.model_state.candidate_models
    ):
        result.blocking_failures.append("model_state.best_model_name is not in candidate_models")
    if not state.model_state.validation_metrics:
        result.warnings.append("model_state.validation_metrics is empty")
    if not state.model_state.cv_results:
        result.warnings.append("model_state.cv_results is empty")
    if not state.model_state.baseline_metrics:
        result.warnings.append("model_state.baseline_metrics is empty")

    if has_external_test:
        _require_readable_non_empty_csv(result, workspace_root, prediction_artifact, "prediction_file")
        _check_prediction_columns_and_rows(
            state,
            result,
            workspace_root,
            prediction_artifact,
            target_contract,
        )
    return result


def _mark_state_fields(result: ReviewCheckResult, *fields: str) -> None:
    result.checked_state_fields.extend(fields)


_DATA_CLEANING_RETRYABLE_FALLBACK_REASONS = {
    "execution_plan.success_criteria phase bundle missing or invalid",
    "execution_plan.agent_plan steps missing or invalid",
    "execution_plan.agent_plan contained no executable steps",
}


def _check_data_cleaning_plan_health(
    result: ReviewCheckResult,
    plan_record: dict[str, Any] | None,
) -> None:
    if not isinstance(plan_record, dict):
        return

    plan_source = str(plan_record.get("plan_source") or "").strip().lower()
    fallback_reason = str(plan_record.get("fallback_reason") or "").strip()
    steps = plan_record.get("steps")

    if plan_source != "deterministic_fallback":
        return

    if fallback_reason in _DATA_CLEANING_RETRYABLE_FALLBACK_REASONS:
        result.blocking_failures.append(
            "data_cleaning did not receive an executable planner plan; "
            f"retry required ({fallback_reason})"
        )
        return

    if fallback_reason and not isinstance(steps, list):
        result.blocking_failures.append(
            "data_cleaning planner fallback record is malformed; retry required"
        )


def _require_artifact(state: GraphState, result: ReviewCheckResult, artifact_key: str) -> None:
    result.checked_artifacts.append(artifact_key)
    artifact_path = str(state.artifacts.get(artifact_key, ""))
    if not artifact_path:
        result.blocking_failures.append(f"artifacts.{artifact_key} is empty")
        return
    if not artifact_exists(state.runtime_state.workspace_root, artifact_path):
        result.blocking_failures.append(f"artifact '{artifact_key}' does not exist at {artifact_path}")


def _require_phase_record(
    state: GraphState,
    result: ReviewCheckResult,
    *,
    bucket: str,
    phase: str,
) -> None:
    nested = dict(state.artifacts.get(bucket) or {})
    artifact_path = str(nested.get(phase) or "")
    result.checked_artifacts.append(f"{bucket}.{phase}")
    if not artifact_path:
        result.blocking_failures.append(f"artifacts.{bucket}.{phase} is empty")
        return
    if not artifact_exists(state.runtime_state.workspace_root, artifact_path):
        result.blocking_failures.append(
            f"artifact '{bucket}.{phase}' does not exist at {artifact_path}"
        )


def _read_phase_record(
    state: GraphState,
    *,
    bucket: str,
    phase: str,
) -> dict[str, Any] | None:
    nested = dict(state.artifacts.get(bucket) or {})
    artifact_path = str(nested.get(phase) or "")
    if not artifact_path:
        return None
    resolved = resolve_artifact_path(state.runtime_state.workspace_root, artifact_path)
    if not resolved.exists():
        return None
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _require_file(
    result: ReviewCheckResult,
    workspace_root: str,
    path: str,
    label: str,
) -> None:
    if not path:
        result.blocking_failures.append(f"{label} is empty")
        return
    if not artifact_exists(workspace_root, path):
        result.blocking_failures.append(f"{label} does not exist at {path}")


def _require_readable_non_empty_csv(
    result: ReviewCheckResult,
    workspace_root: str,
    path: str,
    label: str,
) -> None:
    shape = csv_shape(workspace_root, path)
    if shape is None:
        result.blocking_failures.append(f"{label} is not a readable CSV at {path}")
        return
    result.details[f"{label}_shape"] = list(shape)
    if shape[0] <= 0 or shape[1] <= 0:
        result.blocking_failures.append(f"{label} is empty; shape={shape}")


def _record_csv_details(
    result: ReviewCheckResult,
    label: str,
    workspace_root: str,
    path: str,
) -> None:
    if not path:
        return
    shape = csv_shape(workspace_root, path)
    if shape is not None:
        result.details[f"{label}_shape"] = list(shape)


def _resolve_target_contract(state: GraphState) -> _ResolvedTargetContract:
    target_columns = normalize_target_columns(state.target_spec.target_columns)
    used_legacy_fallback = False
    if not target_columns:
        mirror_columns = normalize_target_columns(
            list(state.problem_understanding.target_columns or [])
            or list(state.data_state.target_columns or [])
            or list(state.task_config.target_columns or [])
        )
        mirror_single = (
            state.problem_understanding.target_column
            or state.data_state.target_column
            or state.task_config.target_column
        )
        _, target_columns = canonical_target_scope(
            target_column=mirror_single,
            target_columns=mirror_columns,
        )
        target_columns = normalize_target_columns(target_columns)
        used_legacy_fallback = bool(target_columns)

    id_columns = normalize_target_columns(state.target_spec.submission.id_columns)
    output_columns = normalize_target_columns(state.target_spec.submission.output_columns)
    if not output_columns:
        output_columns = list(target_columns)

    submission_columns = normalize_target_columns(
        state.target_spec.submission.submission_columns
    )
    if not submission_columns:
        submission_columns = normalize_target_columns(id_columns + output_columns)
    if not submission_columns:
        submission_columns = list(output_columns)

    return _ResolvedTargetContract(
        target_columns=target_columns,
        id_columns=id_columns,
        output_columns=output_columns,
        submission_columns=submission_columns,
        target_mode=str(state.target_spec.target_mode or ""),
        task_family=str(state.target_spec.task_family or ""),
        prediction_kind=str(state.target_spec.prediction_kind or ""),
        used_legacy_fallback=used_legacy_fallback,
    )


def _record_target_contract(
    result: ReviewCheckResult,
    target_contract: _ResolvedTargetContract,
) -> None:
    result.details["target_contract"] = {
        "target_columns": list(target_contract.target_columns),
        "id_columns": list(target_contract.id_columns),
        "output_columns": list(target_contract.output_columns),
        "submission_columns": list(target_contract.submission_columns),
        "target_mode": target_contract.target_mode,
        "task_family": target_contract.task_family,
        "prediction_kind": target_contract.prediction_kind,
    }
    if target_contract.used_legacy_fallback:
        result.warnings.append(
            "target_spec is empty; reviewer fell back to legacy target mirrors"
        )


def _has_external_test(state: GraphState) -> bool:
    return bool(
        state.task_config.dataset_paths.test
        or state.data_state.raw_test_path
        or state.data_state.current_test_path
        or state.feature_state.engineered_test_path
    )


def _same_path(workspace_root: str, left: str, right: str) -> bool:
    if not left or not right:
        return False
    return resolve_artifact_path(workspace_root, left) == resolve_artifact_path(workspace_root, right)


def _feature_columns(columns: list[str], excluded_columns: set[str]) -> list[str]:
    return [column for column in columns if column not in excluded_columns]


def _warn_missing_profile_keys(result: ReviewCheckResult, eda_findings: dict[str, Any]) -> None:
    required_context_keys = {"dataset_shape", "columns"}
    missing = sorted(required_context_keys - set(eda_findings.keys()))
    if missing:
        result.warnings.append(f"EDA profile is missing recommended keys: {missing}")


def _check_prediction_columns_and_rows(
    state: GraphState,
    result: ReviewCheckResult,
    workspace_root: str,
    prediction_artifact: str,
    target_contract: _ResolvedTargetContract,
) -> None:
    if not target_contract.target_columns:
        result.blocking_failures.append(
            "target contract unresolved: reviewer cannot validate prediction_file"
        )
        return

    prediction_columns = csv_columns(workspace_root, prediction_artifact)
    if not prediction_columns:
        return

    expected_submission = list(target_contract.submission_columns)
    result.details["prediction_columns"] = list(prediction_columns)
    result.details["expected_submission_columns"] = list(expected_submission)
    if expected_submission and prediction_columns != expected_submission:
        result.blocking_failures.append(
            "prediction_file columns do not match target_spec submission contract; "
            f"expected={expected_submission}, actual={prediction_columns}"
        )

    prediction_df = _read_csv(workspace_root, prediction_artifact)
    if prediction_df is not None and expected_submission:
        null_columns = [
            column
            for column in expected_submission
            if column in prediction_df.columns and prediction_df[column].isna().any()
        ]
        if null_columns:
            result.blocking_failures.append(
                "prediction_file contains null values in required columns "
                + ", ".join(null_columns)
            )

    expected_path = (
        state.feature_state.engineered_test_path
        or state.data_state.current_test_path
        or state.task_config.dataset_paths.test
    )
    expected_shape = csv_shape(workspace_root, expected_path)
    prediction_shape = csv_shape(workspace_root, prediction_artifact)
    if expected_shape and prediction_shape and expected_shape[0] != prediction_shape[0]:
        result.blocking_failures.append(
            f"prediction row count {prediction_shape[0]} does not match expected test row count {expected_shape[0]}"
        )


def _read_csv(workspace_root: str, artifact_path: str) -> pd.DataFrame | None:
    if not artifact_path:
        return None
    try:
        return pd.read_csv(resolve_artifact_path(workspace_root, artifact_path))
    except Exception:
        return None
