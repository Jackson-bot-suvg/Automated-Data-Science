"""Developer node placeholder for data and modeling phases."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.schemas.state_schema import GraphState
from app.tools.indepth_eda_tools import target_analysis, feature_feature_relationships, outlier_analysis, segment_analysis
from app.tools import file_tools
from app.tools.phase_trace import (
    phase_record_path,
    register_phase_artifact,
    resolve_phase_plan,
    write_phase_record,
)
from app.tools.run_manifest import write_run_manifest
from app.utils.target_scope import get_id_columns, get_target_columns
from app.utils.task_contract import normalize_task_family

_INDEPTH_EDA_REPORT_KEY = "indepth_eda_report"



async def run_indepth_eda(state: GraphState) -> dict:
    """Execute In-Depth EDA phase. update state and artifacts."""
    cleaned_train_path = state.data_state.current_train_path
    eda_findings = state.data_state.eda_findings
    task_id = state.task_config.task_id
    target_columns = get_target_columns(state)
    id_columns = get_id_columns(state)
    model_type = normalize_task_family(
        state.target_spec.task_family or state.problem_understanding.predicted_task_type,
        target_mode=state.target_spec.target_mode,
        target_count=len(target_columns),
    )
    data_quality_flags: list[str] = list(state.data_state.data_quality_flags)

    """Step 1. Validate input"""
    if not cleaned_train_path or not Path(cleaned_train_path).exists():
        return _error_envelope(
            phase="indepth_eda",
            error_type="artifact_missing",
            message=f"cleaned train file not found: '{cleaned_train_path}'",
        )

    if not eda_findings:
        return _error_envelope(
            phase="indepth_eda",
            error_type="artifact_missing",
            message=f"EDA findings not found",
        )

    """Step 2. Read data"""
    df_train_cleaned = pd.read_csv(cleaned_train_path)


    """ Step 3. Load current phase planner plan """
    target_spec_summary = {
        "target_columns": target_columns,
        "id_columns": id_columns,
        "task_family": state.target_spec.task_family,
        "target_mode": state.target_spec.target_mode,
    }
    plan_obj, plan_meta = resolve_phase_plan(
        state,
        phase="indepth_eda",
        fallback_plan_name="indepth_eda_deterministic_fallback",
    )
    if plan_meta["plan_source"] != "planner_execution_plan":
        data_quality_flags.append(
            "In-depth EDA used deterministic fallback plan: "
            + str(plan_meta.get("fallback_reason") or "missing planner plan")
        )
    
    report_path = str(
        state.artifacts.get(_INDEPTH_EDA_REPORT_KEY) or "artifacts/eda/in_depth_eda.md"
    )
    os.makedirs(Path(report_path).parent, exist_ok=True)

    """ Step 4. Execute the agent plan"""
    report = {}
    new_figures: list[str] = []

    import logging as _logging
    _eda_logger = _logging.getLogger(__name__)

    for step in plan_obj.steps:
        tool_name = step.tool
        step_params = dict(step.parameters)

        # ---------------- Target analysis ----------------
        if tool_name == "target_analysis":
            report["target_analysis"] = target_analysis(df_train_cleaned, target_columns, model_type)

            for target_column in target_columns:
                if target_column in df_train_cleaned.columns:
                    if re.search(r"classification", str(model_type), re.IGNORECASE):
                        fig, ax = plt.subplots(figsize=(10, 5))
                        df_train_cleaned[target_column] = df_train_cleaned[target_column].astype(str)
                        sns.countplot(data=df_train_cleaned, x=target_column, ax=ax)
                        path = file_tools.figure_path(
                            workspace_root=state.runtime_state.workspace_root,
                            artifact_root=state.runtime_state.artifact_root,
                            filename=f"target_analysis_{target_column}_{task_id}.png",
                        )
                        fig.savefig(path, dpi=100)
                        plt.close(fig)
                        new_figures.append(path)
                    ## Plot for target analysis        
                    if re.search(r"regression", str(model_type), re.IGNORECASE):
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.histplot(x=df_train_cleaned[target_column], ax=ax)
                        path = file_tools.figure_path(
                            workspace_root=state.runtime_state.workspace_root,
                            artifact_root=state.runtime_state.artifact_root,
                            filename=f"target_analysis_{target_column}_{task_id}.png",
                        )
                        fig.savefig(path, dpi=100)
                        plt.close(fig)
                        new_figures.append(path)
            #print("target analysis done")

        # ---------------- Feature-feature correlations ----------------
        if tool_name == "feature_feature_relationships":
            result = feature_feature_relationships(
                df_train_cleaned,
                target_columns,
                exclude_columns=id_columns,
            )
            report["feature_feature_relationships"] = result
            # Plot for feature-feature correlations
            if "note" not in result.keys():
                corr_matrix = pd.DataFrame(result)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, cmap="coolwarm", center=0, ax=ax)
                path = file_tools.figure_path(
                    workspace_root=state.runtime_state.workspace_root,
                    artifact_root=state.runtime_state.artifact_root,
                    filename=f"correlation_matrix_{task_id}.png",
                )
                fig.savefig(path, dpi=100)
                plt.close(fig)
                new_figures.append(path)
            #print("feature-feature correlations done")

        # ---------------- Outlier analysis ----------------
        if tool_name == "outlier_analysis":
            report["outlier_analysis"] = outlier_analysis(
                df_train_cleaned,
                target_columns,
                exclude_columns=id_columns,
            )

            excluded_numeric = set(target_columns) | set(id_columns)
            num_cols = [
                col for col in df_train_cleaned.select_dtypes(include="number").columns
                if col not in excluded_numeric
            ]
            # Plot for outlier analysis
            if len(num_cols) > 0:
                fig, ax = plt.subplots(figsize=(max(8, len(num_cols)*1.2), 6))
                sns.boxplot(
                    data=df_train_cleaned[num_cols],
                    ax=ax
                    )
                ax.set_title("Boxplot of numeric features")
                ax.set_xlabel("Numeric features")
                ax.set_ylabel("Values")
                ax.tick_params(axis='x', labelrotation=45)
                fig.tight_layout()
                path = file_tools.figure_path(
                    workspace_root=state.runtime_state.workspace_root,
                    artifact_root=state.runtime_state.artifact_root,
                    filename=f"outlier_analysis_{task_id}.png",
                )
                fig.savefig(path, dpi=100)
                plt.close(fig)
                new_figures.append(path)
            #print("outlier analysis done")

        # ---------------- Segmentation analysis ----------------
        if tool_name == "segment_analysis":
            try:
                if "segment_by" in step_params:
                    segments = step_params["segment_by"]
                    report["segment_analysis"] = segment_analysis(df_train_cleaned, target_columns, segments, model_type)
                elif "feature_columns" in step_params:
                    segments = step_params["feature_columns"]
                    report["segment_analysis"] = segment_analysis(df_train_cleaned, target_columns, segments, model_type)
            except Exception as _exc:
                _eda_logger.warning("segment_analysis failed, skipping: %s", _exc)
                report["segment_analysis"] = {"error": str(_exc), "note": "segment_analysis skipped due to error"}
        

    if not report:
        report["Indepth_EDA_fallback"] = {
            "note": "In-depth EDA planner returned no executable steps; preserved preliminary EDA as the available analysis context.",
            "target_columns": target_columns,
            "model_type": model_type,
            "stats": df_train_cleaned.describe(include="all").to_dict()
        }

    plan_record = {
        "phase": "indepth_eda",
        "plan_name": plan_obj.plan_name,
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
            for step in plan_obj.steps
        ],
    }
    plan_record_file = phase_record_path(
        workspace_root=state.runtime_state.workspace_root,
        artifact_root=state.runtime_state.artifact_root,
        category="phase_plans",
        phase="indepth_eda",
    )
    write_phase_record(plan_record_file, plan_record)
        
    existing_figures: list[str] = list(state.artifacts.get("figures", []))

    _write_in_depth_eda_report(report, new_figures, report_path)
    updated_artifacts = register_phase_artifact(
        {
            **state.artifacts,
            "figures": existing_figures + new_figures,
            "indepth_eda_report": report_path,
        },
        bucket="phase_plan_records",
        phase="indepth_eda",
        path=plan_record_file,
    )
    updated_data_state = {
        **state.data_state.model_dump(mode="python"),
        "indepth_eda_summary": report,
        "data_quality_flags": data_quality_flags,
    }
    manifest_path = str(state.artifacts.get("run_manifest") or "")
    if manifest_path:
        write_run_manifest(
            workspace_root=state.runtime_state.workspace_root,
            manifest_path=manifest_path,
            run_id=state.runtime_state.run_id or state.task_config.task_id,
            thread_id=state.runtime_state.thread_id,
            current_phase="indepth_eda",
            phase_history=list(state.phase_history or []) + ["indepth_eda"],
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
        "current_phase": "indepth_eda",
        "phase_history": ["indepth_eda"],
    }


def _build_indepth_eda_payload(df, eda_findings: dict) -> dict:
    return {
        "preliminary_eda_report": eda_findings,
        "sample_rows": df.head(10).to_dict(orient="records")
    }

def _error_envelope(phase: str, error_type: str, message: str, details: dict | None = None) -> dict:
    error_str = f"[{phase}] {error_type}: {message}"
    return {
        "runtime_state": {
            "interrupted": True,
        },
        "review_state": {
            "fatal_errors": [error_str]
        },
    }

def _write_in_depth_eda_report(profile_json: dict, figures: list[str], report_path: str) -> None:
    lines: list[str] = [
    "# In-Depth EDA Report\n\n",
    f"This report is generated by the Indepth EDA tool and will be submitted in its entirety to the LLM Planner for tool invocation.\n\n",
    "```json\n" + json.dumps(profile_json, ensure_ascii=False, indent=2) + "\n```\n\n"
    ]
    if figures:
        lines.append("\n## Generated Figures\n\n")
        for fig_path in figures:
            lines.append(f"- `{fig_path}`\n\n")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
