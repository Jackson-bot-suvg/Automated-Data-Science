"""Feature Engineering node.

Uses the LLM Planner to generate a feature tool call chain from the latest
post-cleaning profile and automatically applies the feature toolset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

import app.tools.feature_tools as feature_tools
from app.schemas.state_schema import GraphState
from app.tools import file_tools
from app.tools.phase_trace import (
    phase_record_path,
    register_phase_artifact,
    resolve_phase_plan,
    write_phase_record,
)
from app.tools.run_manifest import write_run_manifest
from app.utils.target_scope import get_target_columns, get_id_columns

# ──────────────────────────────────────────────
# Reserved artifact keys
# ──────────────────────────────────────────────
_ENGINEERED_TRAIN_KEY = "engineered_train_data"
_ENGINEERED_TEST_KEY = "engineered_test_data"
_FEATURE_SUMMARY_KEY = "feature_summary"


async def run_feature_engineering(state: GraphState) -> dict:
    """Execute the Feature Engineering phase and return a partial state update."""
    train_path = state.data_state.current_train_path
    test_path = state.data_state.current_test_path

    # 1. Validate input paths
    if not train_path or not Path(train_path).exists():
        return _error_envelope(
            phase="feature_engineering",
            error_type="artifact_missing",
            message=f"Cleaned training dataset does not exist: '{train_path}'",
        )

    initial_data_quality_flags: list[str] = list(state.data_state.data_quality_flags)

    try:
        df_train = pd.read_csv(train_path)
    except EmptyDataError:
        return _error_envelope(
            phase="feature_engineering",
            error_type="artifact_invalid",
            message=f"Cleaned training dataset is empty or has no parseable columns: '{train_path}'",
        )
    df_test = pd.DataFrame()
    if test_path and Path(test_path).exists():
        try:
            df_test = pd.read_csv(test_path)
        except EmptyDataError:
            initial_data_quality_flags.append(
                f"Feature engineering skipped unreadable empty test artifact: {test_path}"
            )
            df_test = pd.DataFrame()

    # Get target columns and ID columns from target_spec
    target_cols = get_target_columns(state)
    id_cols = get_id_columns(state)
    protected_cols = set(target_cols) | set(id_cols)
    original_feature_columns = [
        column for column in df_train.columns if column not in protected_cols
    ]

    # Build target_spec_summary
    target_spec_summary = {
        "target_columns": target_cols,
        "id_columns": id_cols,
        "task_family": state.target_spec.task_family,
        "target_mode": state.target_spec.target_mode,
    }

    # 2. Rebuild the latest profile from cleaned data for the Planner
    feature_transform_summary: dict[str, Any] = {}
    data_quality_flags: list[str] = list(initial_data_quality_flags)
    leakage_checks: list[str] = []

    # 3. Read the planner plan for the current phase
    plan, plan_meta = resolve_phase_plan(
        state,
        phase="feature_engineering",
        fallback_plan_name="feature_engineering_deterministic_fallback",
    )
    if plan_meta["plan_source"] != "planner_execution_plan":
        feature_transform_summary["planner_fallback"] = {
            "reasoning": "Feature engineering used deterministic fallback and preserved input features.",
            "params": {"reason": plan_meta.get("fallback_reason")},
        }

    # 4. Dynamically dispatch and execute tool calls with target protection
    for step in plan.steps:
        tool_func_name = step.tool
        params = dict(step.parameters)

        func = getattr(feature_tools, tool_func_name, None)
        if not callable(func):
            data_quality_flags.append(
                f"Feature engineering skipped unknown tool '{tool_func_name}'"
            )
            feature_transform_summary[f"Step: {tool_func_name} (skipped)"] = {
                "reasoning": step.reasoning,
                "params": params,
                "note": "Tool does not exist; skipped",
            }
            continue

        # Target protection: encode_categorical must not encode target columns
        if tool_func_name == "encode_categorical" and "column" in params:
            if params["column"] in protected_cols:
                data_quality_flags.append(
                    f"Target protection skipped encoding on protected column '{params['column']}'"
                )
                feature_transform_summary[f"Step: {tool_func_name} (skipped)"] = {
                    "reasoning": step.reasoning,
                    "params": params,
                    "note": f"'{params['column']}' is a protected column; skipped",
                }
                continue

        # Target protection: group_rare_categories must not modify target columns
        if tool_func_name == "group_rare_categories" and "column" in params:
            if params["column"] in protected_cols:
                data_quality_flags.append(
                    f"Target protection skipped rare-category grouping on protected column '{params['column']}'"
                )
                continue

        # Target protection: create_binning must not bin target columns
        if tool_func_name == "create_binning" and "column" in params:
            if params["column"] in protected_cols:
                data_quality_flags.append(
                    f"Target protection skipped binning on protected column '{params['column']}'"
                )
                continue

        if tool_func_name == "extract_regex":
            if params.get("column") in protected_cols or params.get("new_col") in protected_cols:
                data_quality_flags.append(
                    f"Target protection skipped extract_regex on protected columns {params.get('column')} -> {params.get('new_col')}"
                )
                continue

        if tool_func_name == "create_interaction":
            cols = params.get("cols") or []
            if any(col in protected_cols for col in cols) or params.get("new_col") in protected_cols:
                data_quality_flags.append(
                    f"Target protection skipped create_interaction on protected columns {cols}"
                )
                continue

        feature_transform_summary[f"Step: {tool_func_name}"] = {
            "reasoning": step.reasoning,
            "params": params,
        }

        try:
            df_train = func(df_train, **params)
            if not df_test.empty:
                df_test = func(df_test, **params)
        except Exception as e:
            return _error_envelope(
                phase="feature_engineering",
                error_type="tool_execution_failed",
                message=f"Executing feature tool '{tool_func_name}' failed: {str(e)}",
            )

    # 5. Build selected_features excluding target and ID columns
    excluded_cols = set(target_cols) | set(id_cols)
    selected_features = [c for c in df_train.columns if c not in excluded_cols]
    if not selected_features:
        return _error_envelope(
            phase="feature_engineering",
            error_type="schema_error",
            message="No usable feature columns remain after excluding target and id columns",
        )

    # Leakage checks
    target_in_features = [c for c in target_cols if c in selected_features]
    if target_in_features:
        leakage_checks.append(
            f"WARNING: target columns leaked into selected_features: {target_in_features}"
        )
    else:
        leakage_checks.append("PASS: target columns excluded from selected_features")

    id_in_features = [c for c in id_cols if c in selected_features]
    if id_in_features:
        leakage_checks.append(
            f"WARNING: id columns present in selected_features: {id_in_features}"
        )
    else:
        leakage_checks.append("PASS: id columns excluded from selected_features")

    # 6. Align train/test feature schemas
    if not df_test.empty:
        train_feature_cols = sorted(set(df_train.columns) - excluded_cols)
        test_feature_cols = sorted(set(df_test.columns) - excluded_cols)

        if set(train_feature_cols) != set(test_feature_cols):
            missing_in_test = sorted(set(train_feature_cols) - set(test_feature_cols))
            extra_in_test = sorted(set(test_feature_cols) - set(train_feature_cols))

            if missing_in_test:
                for col in missing_in_test:
                    df_test[col] = 0
                data_quality_flags.append(
                    f"Schema alignment filled missing test columns with zeros: {missing_in_test}"
                )
            if extra_in_test:
                df_test = df_test.drop(columns=extra_in_test)
                data_quality_flags.append(
                    f"Schema alignment dropped extra test columns: {extra_in_test}"
                )
            leakage_checks.append(
                "INFO: schema alignment applied between train and test features "
                f"(missing_in_test={missing_in_test}, extra_in_test={extra_in_test})"
            )
        else:
            leakage_checks.append("PASS: train/test feature schema aligned")

        ordered_cols = [column for column in df_train.columns if column in df_test.columns]
        remaining_test_only = [column for column in df_test.columns if column not in ordered_cols]
        df_test = df_test[ordered_cols + remaining_test_only]

    # 7. Prepare output persistence
    engineered_train_path = str(
        state.artifacts.get(_ENGINEERED_TRAIN_KEY) or "artifacts/features/engineered_train.csv"
    )
    engineered_test_target = str(
        state.artifacts.get(_ENGINEERED_TEST_KEY) or "artifacts/features/engineered_test.csv"
    )
    feature_summary_path = str(
        state.artifacts.get(_FEATURE_SUMMARY_KEY) or "artifacts/features/feature_summary.json"
    )
    file_tools.ensure_artifact_parent(engineered_train_path)
    df_train.to_csv(engineered_train_path, index=False)
    engineered_test_path = ""
    if not df_test.empty:
        file_tools.ensure_artifact_parent(engineered_test_target)
        df_test.to_csv(engineered_test_target, index=False)
        engineered_test_path = engineered_test_target

    # Write JSON summary
    feature_summary_data = {
        "plan_name": plan.plan_name,
        "transformations": feature_transform_summary,
        "selected_features": selected_features,
        "target_columns": target_cols,
        "id_columns": id_cols,
        "leakage_checks": leakage_checks,
        "engineered_train_shape": list(df_train.shape),
        "engineered_test_shape": list(df_test.shape) if not df_test.empty else [],
    }
    file_tools.ensure_artifact_parent(feature_summary_path)
    with open(feature_summary_path, "w", encoding="utf-8") as fh:
        json.dump(feature_summary_data, fh, ensure_ascii=False, indent=2)

    updated_feature_state = {
        **state.feature_state.model_dump(),
        "engineered_train_path": engineered_train_path,
        "engineered_test_path": engineered_test_path,
        "selected_features": selected_features,
        "dropped_features": sorted(
            set(original_feature_columns) - set(selected_features)
        ),
        "feature_transform_summary": feature_transform_summary,
        "leakage_checks": leakage_checks,
    }

    plan_record = {
        "phase": "feature_engineering",
        "plan_name": plan.plan_name,
        "plan_source": plan_meta.get("plan_source"),
        "fallback_reason": plan_meta.get("fallback_reason"),
        "criteria": plan_meta.get("criteria") or [],
        "allowed_tools": plan_meta.get("allowed_tools") or [],
        "target_spec_summary": target_spec_summary,
        "steps": [
            {
                "tool": step.tool,
                "parameters": dict(step.parameters),
                "reasoning": step.reasoning,
            }
            for step in plan.steps
        ],
    }
    plan_record_file = phase_record_path(
        workspace_root=state.runtime_state.workspace_root,
        artifact_root=state.runtime_state.artifact_root,
        category="phase_plans",
        phase="feature_engineering",
    )
    write_phase_record(plan_record_file, plan_record)

    manifest_path = str(state.artifacts.get("run_manifest") or "")
    if manifest_path:
        write_run_manifest(
            workspace_root=state.runtime_state.workspace_root,
            manifest_path=manifest_path,
            run_id=state.runtime_state.run_id or state.task_config.task_id,
            thread_id=state.runtime_state.thread_id,
            current_phase="feature_engineering",
            phase_history=list(state.phase_history or []) + ["feature_engineering"],
            task_config=state.task_config.model_dump(mode="python"),
            target_spec=state.target_spec.model_dump(mode="python"),
            artifacts={
                **state.artifacts,
                "feature_summary": feature_summary_path,
                "engineered_train_data": engineered_train_path,
                "engineered_test_data": engineered_test_path,
                "phase_plan_records": {
                    **dict(state.artifacts.get("phase_plan_records") or {}),
                    "feature_engineering": plan_record_file,
                },
            },
            runtime_state=state.runtime_state.model_dump(mode="python"),
            data_state={
                **state.data_state.model_dump(mode="python"),
                "data_quality_flags": data_quality_flags,
            },
            feature_state=updated_feature_state,
            review_state=state.review_state.model_dump(mode="python"),
            report_state=state.report_state.model_dump(mode="python"),
        )

    return {
        "feature_state": updated_feature_state,
        "data_state": {
            **state.data_state.model_dump(),
            "data_quality_flags": data_quality_flags,
        },
        "artifacts": register_phase_artifact(
            {
                **state.artifacts,
            "feature_summary": feature_summary_path,
            "engineered_train_data": engineered_train_path,
            "engineered_test_data": engineered_test_path,
            "run_manifest": manifest_path or state.artifacts.get("run_manifest", ""),
            },
            bucket="phase_plan_records",
            phase="feature_engineering",
            path=plan_record_file,
        ),
        "current_phase": "feature_engineering",
        "phase_history": ["feature_engineering"],
    }


# ──────────────────────────────────────────────
# Internal helper functions
# ──────────────────────────────────────────────


def _error_envelope(
    phase: str, error_type: str, message: str, details: dict | None = None
) -> dict:
    error_str = f"[{phase}] {error_type}: {message}"
    return {
        "runtime_state": {"interrupted": True},
        "review_state": {"fatal_errors": [error_str]},
    }
