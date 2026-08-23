"""Data Cleaning node.

Uses the LLM Planner to generate cleaning tool calls from the EDA profile,
then dynamically maps and applies the underlying ``cleaning_tools`` helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

import app.tools.cleaning_tools as cleaning_tools
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
_CLEANED_TRAIN_KEY = "cleaned_train_data"
_CLEANED_TEST_KEY = "cleaned_test_data"
_CLEANING_REPORT_KEY = "cleaning_report"


async def run_data_cleaning(state: GraphState) -> dict:
    """Execute the Data Cleaning phase and return a partial state update."""
    train_path = state.data_state.current_train_path
    test_path = state.data_state.current_test_path
    eda_findings = state.data_state.eda_findings

    # 1. Validate paths and required context
    if not train_path or not Path(train_path).exists():
        return _error_envelope(
            phase="data_cleaning",
            error_type="artifact_missing",
            message=f"Training dataset file does not exist: '{train_path}'",
        )
    if not eda_findings:
        return _error_envelope(
            phase="data_cleaning",
            error_type="missing_context",
            message="EDA profile JSON was not found",
        )

    df_train = pd.read_csv(train_path)
    df_test = (
        pd.read_csv(test_path)
        if test_path and Path(test_path).exists()
        else pd.DataFrame()
    )

    cleaning_summary: dict[str, Any] = {}
    data_quality_flags: list[str] = list(state.data_state.data_quality_flags)

    # Get target columns and ID columns from target_spec
    target_cols = get_target_columns(state)
    id_cols = get_id_columns(state)
    protected_cols = set(target_cols) | set(id_cols)

    # Build target_spec_summary for the LLM Planner
    target_spec_summary = {
        "target_columns": target_cols,
        "id_columns": id_cols,
        "task_family": state.target_spec.task_family,
        "target_mode": state.target_spec.target_mode,
    }

    # 2. Read the planner plan for the current phase
    plan, plan_meta = resolve_phase_plan(
        state,
        phase="data_cleaning",
        fallback_plan_name="data_cleaning_deterministic_fallback",
    )
    if plan_meta["plan_source"] != "planner_execution_plan":
        data_quality_flags.append(
            "Data cleaning used deterministic fallback plan: "
            + str(plan_meta.get("fallback_reason") or "missing planner plan")
        )
    elif not plan.steps and plan_meta.get("fallback_reason") == "no_action_needed":
        cleaning_summary["no_action_needed"] = {
            "reason": "Planner determined no cleaning action was required for the current dataset snapshot.",
            "plan_name": plan.plan_name,
        }

    # 3. Dynamically apply tool calls with target-column protection
    for step in plan.steps:
        tool_func_name = step.tool
        params = dict(step.parameters)  # Copy to avoid mutating the original object

        func = getattr(cleaning_tools, tool_func_name, None)
        if not callable(func):
            data_quality_flags.append(
                f"Data cleaning skipped unknown tool '{tool_func_name}'"
            )
            cleaning_summary[f"Step: {tool_func_name} (skipped)"] = {
                "reasoning": step.reasoning,
                "params": params,
                "note": "Tool does not exist; skipped",
            }
            continue

        # Target protection: automatically exclude target and ID columns from drop_columns
        if tool_func_name == "drop_columns" and "columns" in params:
            original_cols = params["columns"]
            if isinstance(original_cols, str):
                original_cols = [original_cols]
            safe_cols = [c for c in original_cols if c not in protected_cols]
            if len(safe_cols) < len(original_cols):
                removed = set(original_cols) - set(safe_cols)
                data_quality_flags.append(
                    f"Target protection removed protected columns from drop_columns: {removed}"
                )
            params["columns"] = safe_cols
            if not safe_cols:
                cleaning_summary[f"Step: {tool_func_name} (skipped)"] = {
                    "reasoning": step.reasoning,
                    "params": params,
                    "note": "All requested columns are protected; skipped",
                }
                continue

        # Record execution details for each step
        cleaning_summary[f"Step: {tool_func_name}"] = {
            "reasoning": step.reasoning,
            "params": params,
        }

        try:
            df_train = func(df_train, **params)
            if not df_test.empty:
                df_test = func(df_test, **params)
        except Exception as e:
            return _error_envelope(
                phase="data_cleaning",
                error_type="tool_execution_failed",
                message=f"Executing tool '{tool_func_name}' failed: {str(e)}",
            )

    # 4. Post-cleaning validation
    # 4a. Validate that the training dataset still contains all target columns
    if target_cols:
        missing_targets = [c for c in target_cols if c not in df_train.columns]
        if missing_targets:
            return _error_envelope(
                phase="data_cleaning",
                error_type="target_column_lost",
                message=f"Training dataset lost target columns after cleaning: {missing_targets}",
            )

    # 4b. Ensure the test dataset does not contain target columns
    if not df_test.empty and target_cols:
        leaked_targets = [c for c in target_cols if c in df_test.columns]
        if leaked_targets:
            df_test = df_test.drop(columns=leaked_targets)
            data_quality_flags.append(
                f"Target columns were found and removed from the test dataset: {leaked_targets}"
            )

    # 5. Ensure directories exist and persist outputs
    cleaned_train_path = str(state.artifacts.get(_CLEANED_TRAIN_KEY) or "artifacts/data/cleaned_train.csv")
    cleaned_test_target = str(state.artifacts.get(_CLEANED_TEST_KEY) or "artifacts/data/cleaned_test.csv")
    cleaning_report_path = str(state.artifacts.get(_CLEANING_REPORT_KEY) or "artifacts/data/cleaning_report.md")
    file_tools.ensure_artifact_parent(cleaned_train_path)
    df_train.to_csv(cleaned_train_path, index=False)
    cleaned_test_path = ""
    if not df_test.empty:
        file_tools.ensure_artifact_parent(cleaned_test_target)
        df_test.to_csv(cleaned_test_target, index=False)
        cleaned_test_path = cleaned_test_target

    _write_cleaning_report(
        cleaning_summary,
        plan.plan_name,
        df_train,
        df_test,
        cleaning_report_path,
    )

    plan_record = {
        "phase": "data_cleaning",
        "plan_name": plan.plan_name,
        "plan_source": plan_meta.get("plan_source"),
        "fallback_reason": plan_meta.get("fallback_reason"),
        "criteria": plan_meta.get("criteria") or [],
        "allowed_tools": plan_meta.get("allowed_tools") or [],
        "steps": [
            {
                "tool": step.tool,
                "parameters": dict(step.parameters),
                "reasoning": step.reasoning,
            }
            for step in plan.steps
        ],
    }
    plan_record_path = phase_record_path(
        workspace_root=state.runtime_state.workspace_root,
        artifact_root=state.runtime_state.artifact_root,
        category="phase_plans",
        phase="data_cleaning",
    )
    write_phase_record(plan_record_path, plan_record)

    updated_data_state = {
        **state.data_state.model_dump(),
        "current_train_path": cleaned_train_path,
        "current_test_path": cleaned_test_path,
        "cleaning_summary": cleaning_summary,
        "data_quality_flags": data_quality_flags,
    }
    updated_artifacts = register_phase_artifact(
        {
            **state.artifacts,
            "cleaned_train_data": cleaned_train_path,
            "cleaned_test_data": cleaned_test_path,
            "cleaning_report": cleaning_report_path,
        },
        bucket="phase_plan_records",
        phase="data_cleaning",
        path=plan_record_path,
    )
    manifest_path = str(state.artifacts.get("run_manifest") or "")
    if manifest_path:
        write_run_manifest(
            workspace_root=state.runtime_state.workspace_root,
            manifest_path=manifest_path,
            run_id=state.runtime_state.run_id or state.task_config.task_id,
            thread_id=state.runtime_state.thread_id,
            current_phase="data_cleaning",
            phase_history=list(state.phase_history or []) + ["data_cleaning"],
            task_config=state.task_config.model_dump(mode="python"),
            target_spec=state.target_spec.model_dump(mode="python"),
            artifacts=updated_artifacts,
            runtime_state=state.runtime_state.model_dump(mode="python"),
            data_state=updated_data_state,
            feature_state=state.feature_state.model_dump(mode="python"),
            review_state=state.review_state.model_dump(mode="python"),
            report_state=state.report_state.model_dump(mode="python"),
        )

    return {
        "data_state": updated_data_state,
        "artifacts": updated_artifacts,
        "current_phase": "data_cleaning",
        "phase_history": ["data_cleaning"],
    }


# ──────────────────────────────────────────────
# Internal helper functions
# ──────────────────────────────────────────────


def _write_cleaning_report(
    summary: dict[str, Any],
    plan_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    report_path: str,
) -> None:
    lines: list[str] = [
        "# Data Cleaning Report\n\n",
        f"**Executed AI Plan**: {plan_name}\n\n",
        "## Cleaning Tool Execution Chain\n\n",
    ]
    for key, value in summary.items():
        lines.append(f"### {key}\n")
        lines.append(f"- **Reason**: {value.get('reasoning')}\n")
        lines.append(f"- **Parameters**: {value.get('params')}\n\n")

    lines.append("## Post-cleaning Dataset Summary\n\n")
    lines.append(f"- Training shape: {df_train.shape}\n")
    if not df_test.empty:
        lines.append(f"- Test shape: {df_test.shape}\n")
    lines.append(f"- Training columns: {', '.join(df_train.columns.tolist())}\n")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def _error_envelope(
    phase: str, error_type: str, message: str, details: dict | None = None
) -> dict:
    error_str = f"[{phase}] {error_type}: {message}"
    return {
        "runtime_state": {"interrupted": True},
        "review_state": {"fatal_errors": [error_str]},
    }
