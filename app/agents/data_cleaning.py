"""数据清洗（Data Cleaning）节点。

使用 LLM Planner 基于 EDA 画像生成清洗 Tool Calls，
然后动态映射并应用底层 `cleaning_tools` 工具。
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
# 保留工件路径
# ──────────────────────────────────────────────
_CLEANED_TRAIN_KEY = "cleaned_train_data"
_CLEANED_TEST_KEY = "cleaned_test_data"
_CLEANING_REPORT_KEY = "cleaning_report"


async def run_data_cleaning(state: GraphState) -> dict:
    """执行 Data Cleaning 阶段，返回局部 State 更新字典。"""
    train_path = state.data_state.current_train_path
    test_path = state.data_state.current_test_path
    eda_findings = state.data_state.eda_findings

    # ── 1. 路径与 Context 校验 ──────────────────────────────
    if not train_path or not Path(train_path).exists():
        return _error_envelope(
            phase="data_cleaning",
            error_type="artifact_missing",
            message=f"训练集文件不存在: '{train_path}'",
        )
    if not eda_findings:
        return _error_envelope(
            phase="data_cleaning",
            error_type="missing_context",
            message=f"未找到 EDA json 画像",
        )

    df_train = pd.read_csv(train_path)
    df_test = (
        pd.read_csv(test_path)
        if test_path and Path(test_path).exists()
        else pd.DataFrame()
    )

    cleaning_summary: dict[str, Any] = {}
    data_quality_flags: list[str] = list(state.data_state.data_quality_flags)

    # ── 获取目标列和 ID 列（从 target_spec 读取）─────
    target_cols = get_target_columns(state)
    id_cols = get_id_columns(state)
    protected_cols = set(target_cols) | set(id_cols)

    # ── 构建 target_spec_summary 供 LLM Planner 消费 ───
    target_spec_summary = {
        "target_columns": target_cols,
        "id_columns": id_cols,
        "task_family": state.target_spec.task_family,
        "target_mode": state.target_spec.target_mode,
    }

    # ── 2. 读取当前 phase 的 planner plan（无内部二次规划）──────
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

    # ── 3. Tool Calls 动态应用（带目标列保护）───────────
    for step in plan.steps:
        tool_func_name = step.tool
        params = dict(step.parameters)  # 复制一份，避免修改原始对象

        func = getattr(cleaning_tools, tool_func_name, None)
        if not callable(func):
            data_quality_flags.append(
                f"Data cleaning skipped unknown tool '{tool_func_name}'"
            )
            cleaning_summary[f"Step: {tool_func_name} (skipped)"] = {
                "reasoning": step.reasoning,
                "params": params,
                "note": "工具不存在，跳过",
            }
            continue

        # 目标列保护：如果是 drop_columns，自动剔除目标列和 ID 列
        if tool_func_name == "drop_columns" and "columns" in params:
            original_cols = params["columns"]
            if isinstance(original_cols, str):
                original_cols = [original_cols]
            safe_cols = [c for c in original_cols if c not in protected_cols]
            if len(safe_cols) < len(original_cols):
                removed = set(original_cols) - set(safe_cols)
                data_quality_flags.append(
                    f"目标列保护：已从 drop_columns 中剔除受保护列 {removed}"
                )
            params["columns"] = safe_cols
            if not safe_cols:
                cleaning_summary[f"Step: {tool_func_name} (skipped)"] = {
                    "reasoning": step.reasoning,
                    "params": params,
                    "note": "所有列均为受保护列，跳过",
                }
                continue

        # 记录每步执行日志
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
                message=f"执行工具 '{tool_func_name}' 失败: {str(e)}",
            )

    # ── 4. 清洗后验证 ───────────────────────────
    # 4a. 验证训练集仍包含所有目标列
    if target_cols:
        missing_targets = [c for c in target_cols if c not in df_train.columns]
        if missing_targets:
            return _error_envelope(
                phase="data_cleaning",
                error_type="target_column_lost",
                message=f"清洗后训练集丢失目标列: {missing_targets}",
            )

    # 4b. 确保测试集不包含目标列
    if not df_test.empty and target_cols:
        leaked_targets = [c for c in target_cols if c in df_test.columns]
        if leaked_targets:
            df_test = df_test.drop(columns=leaked_targets)
            data_quality_flags.append(
                f"测试集中发现并删除了目标列: {leaked_targets}"
            )

    # ── 5. 确保目录存在并保存 ──────────────────
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
# 内部辅助函数
# ──────────────────────────────────────────────


def _write_cleaning_report(
    summary: dict[str, Any],
    plan_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    report_path: str,
) -> None:
    lines: list[str] = [
        f"# Data Cleaning 智能清洗报告\n\n",
        f"**执行的 AI Plan**: {plan_name}\n\n",
        "## 清洗工具执行链\n\n",
    ]
    for key, value in summary.items():
        lines.append(f"### {key}\n")
        lines.append(f"- **原因**: {value.get('reasoning')}\n")
        lines.append(f"- **参数**: {value.get('params')}\n\n")

    lines.append("## 清洗后数据集概览\n\n")
    lines.append(f"- 训练集形状：{df_train.shape}\n")
    if not df_test.empty:
        lines.append(f"- 测试集形状：{df_test.shape}\n")
    lines.append(f"- 训练集列名：{', '.join(df_train.columns.tolist())}\n")

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
