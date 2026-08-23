"""Prompt assembly for phase-aware agent tool planning."""

from __future__ import annotations

import json
from typing import Any


PHASE_TOOLLIST_SYSTEM_PROMPT = """
You are a phase-level planner that returns only a ToolCallPlan JSON object.
Graph routing is handled outside of planner.
Use only tools from the provided allowlist for the current phase.
Target scope must come from target_spec only.
Return only JSON with keys: plan_name, steps.
"""


def _protected_column_lines(target_spec_summary: dict[str, Any]) -> list[str]:
    target_columns = list(target_spec_summary.get("target_columns") or [])
    id_columns = list(target_spec_summary.get("id_columns") or [])
    protected = target_columns + id_columns
    return [
        f"- Protected target columns: {target_columns}",
        f"- Protected id columns: {id_columns}",
        f"- Never mutate, encode, bin, drop, or derive from protected columns: {protected}",
    ]


def _phase_hard_rules(phase: str) -> list[str]:
    common = [
        "- Use only tools from the allowlist for this phase.",
        "- Do not invent tools, fields, or column names.",
        "- Prefer a safe minimal executable plan over an empty plan when this phase is actionable.",
    ]

    phase_rules = {
        "data_cleaning": [
            "- Never drop protected target or id columns.",
            "- Use drop_columns only for clearly low-value feature columns such as extreme-missingness feature columns.",
            "- Use clip_outliers only on numeric columns.",
            "- Use fill_missing only with a schema-compatible strategy for the observed column type.",
        ],
        "indepth_eda": [
            "- Prefer target-aware analysis, relationship analysis, and outlier analysis before segment analysis.",
            "- Use segment_analysis only with observed non-protected feature columns.",
            "- Do not propose feature-engineering or modeling steps in this phase.",
        ],
        "feature_engineering": [
            "- Never transform protected target or id columns.",
            "- Use extract_regex only on string-like columns with an obvious parseable pattern.",
            "- Use create_binning only on numeric columns with enough distinct values.",
            "- Use create_interaction only when every source column is numeric.",
            "- Use group_rare_categories and encode_categorical only on non-protected categorical columns.",
            "- New feature names must not collide with protected columns.",
        ],
        "model_building": [
            "- Use only configure_* tools for this phase.",
            "- CV fold count and CV strategy are controlled by the modeling module; do not try to configure CV here.",
            "- Keep candidate models conservative, task-compatible, and limited to at most 5 names.",
            "- Scoring must stay aligned with the resolved target contract primary metric.",
            "- Do not emit unsupported model names or binary-only scoring assumptions for non-binary tasks.",
        ],
    }
    return common + phase_rules.get(phase, [])


def _phase_anti_patterns(phase: str) -> list[str]:
    anti_patterns = {
        "data_cleaning": [
            "- Bad: dropping target columns, id columns, or broad feature sets without evidence.",
            "- Bad: clipping or imputing a column that is not present in the profiled schema.",
        ],
        "indepth_eda": [
            "- Bad: using a segment feature that is itself a target or id column.",
            "- Bad: emitting feature transforms or training actions in EDA.",
        ],
        "feature_engineering": [
            "- Bad: encoding the target column as a feature.",
            "- Bad: applying regex extraction to numeric columns or already-split columns.",
            "- Bad: arithmetic interactions over categorical or mixed-type inputs.",
        ],
        "model_building": [
            "- Bad: classification scoring for regression tasks.",
            "- Bad: binary-only scoring assumptions for multilabel tasks.",
            "- Bad: unsupported candidate-model names, more than 3 candidate models, or empty model lists when the phase is actionable.",
        ],
    }
    return anti_patterns.get(phase, [])


def _model_building_task_guidance(
    normalized_task_type: str,
    target_spec_summary: dict[str, Any],
) -> list[str]:
    task_family = str(target_spec_summary.get("task_family") or "").lower()
    primary_metric = str(target_spec_summary.get("primary_metric") or "")
    candidate_metrics = list(target_spec_summary.get("candidate_metrics") or [])

    header = [
        f"- Resolved task_family: {task_family}",
        f"- Preferred primary metric: {primary_metric}",
        f"- Candidate metrics: {candidate_metrics}",
    ]

    if "multilabel" in task_family or normalized_task_type == "multilabel":
        return header + [
            "- Select at most 5 conservative multilabel-compatible candidate models such as LogisticRegression, RandomForest, ExtraTrees, LightGBM, or XGBoost base estimators.",
            "- Keep CV conservative and reproducible; avoid binary-only metric assumptions.",
            "- Prefer multilabel-safe scoring semantics such as f1_micro, f1_macro, or roc_auc_macro_ovr when available in target_spec.",
        ]
    if "multiclass" in task_family:
        return header + [
            "- Select at most 5 multiclass-friendly baselines. Prefer LogisticRegression, RandomForest, ExtraTrees, HistGradientBoosting, and LightGBM when available; use SVC only when label outputs are sufficient.",
            "- Prefer macro-aware scoring semantics such as f1_macro or balanced_accuracy when they are present in target_spec.",
            "- Do not spend plan budget on CV settings; the modeling module owns fold count and strategy.",
        ]
    if "binary" in task_family or normalized_task_type == "classification":
        return header + [
            "- Select at most 5 robust binary baselines. Prefer LogisticRegression, RandomForest, ExtraTrees, HistGradientBoosting, and LightGBM first.",
            "- When prediction_kind implies probability outputs, avoid SVC and prefer probability-friendly models.",
            "- Do not spend plan budget on CV settings; the modeling module owns fold count and strategy.",
        ]
    if "cluster" in task_family or normalized_task_type == "clustering":
        return header + [
            "- Select at most 2 unsupervised clustering candidates such as KMeans and DBSCAN.",
            "- Use clustering-appropriate metrics such as silhouette_score when the target contract provides them.",
            "- Do not emit classification or regression scoring guidance for clustering tasks.",
        ]
    if "regress" in task_family or normalized_task_type == "regression":
        return header + [
            "- Select at most 5 regression-compatible baselines. Prefer Ridge, RandomForest, ExtraTrees, GradientBoosting, and LightGBM first.",
            "- Use regression metrics only; do not reuse classification scoring assumptions.",
            "- Do not spend plan budget on CV settings; the modeling module owns fold count and strategy.",
        ]
    return header


def _self_check_lines() -> list[str]:
    return [
        "- Before emitting JSON, verify every tool is in the current phase allowlist.",
        "- Verify every chosen parameter name matches the declared schema exactly.",
        "- Verify all required fields are present.",
        "- Verify referenced columns exist in the observed schema and match the tool's expected type.",
        "- Verify no step mutates or uses protected target/id columns as features.",
    ]


def _retry_context_lines(retry_context: dict[str, Any] | None) -> list[str]:
    if not isinstance(retry_context, dict) or not retry_context:
        return ["- No retry-specific planner feedback is available for this phase."]

    lines = [
        f"- Retry attempt: {retry_context.get('attempt')}",
        f"- Last review status: {retry_context.get('last_status')}",
        f"- Blocking failures to address: {retry_context.get('blocking_failures') or []}",
        f"- Reviewer warnings: {retry_context.get('warnings') or []}",
        f"- Previous plan name: {retry_context.get('previous_plan_name') or ''}",
        f"- Previous plan steps: {retry_context.get('previous_plan_steps') or []}",
        "- You must address the blocking failures directly and avoid repeating invalid steps from the previous plan.",
    ]
    return lines


def build_phase_tolist_prompt(
    *,
    phase: str,
    normalized_task_type: str,
    profile_summary: dict[str, Any],
    target_spec_summary: dict[str, Any],
    retry_context: dict[str, Any] | None,
    allowed_tools: list[str],
    tool_parameter_schemas: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build a prompt that asks for ToolCallPlan-only output for one phase."""

    tool_parameter_schemas = tool_parameter_schemas or {}
    task_guidance = (
        _model_building_task_guidance(normalized_task_type, target_spec_summary)
        if phase == "model_building"
        else ["- No additional task-family guidance for this phase."]
    )

    return "\n".join(
        [
            "=== PHASE TOOLLIST PLANNING ===",
            "Create one executable ToolCallPlan for the target phase only.",
            "Do not reason about graph routing or other phases.",
            f"target_phase: {phase}",
            f"normalized_task_type: {normalized_task_type}",
            "",
            "Target spec summary:",
            json.dumps(target_spec_summary, ensure_ascii=False, indent=2, default=str),
            "",
            "Protected-column rules:",
            *_protected_column_lines(target_spec_summary),
            "",
            "Phase profile summary:",
            json.dumps(profile_summary, ensure_ascii=False, indent=2, default=str),
            "",
            "Allowed tools (strict, current phase only):",
            json.dumps(allowed_tools, ensure_ascii=False, indent=2),
            "",
            "Tool parameter schemas (strict; do not use undeclared fields):",
            json.dumps(tool_parameter_schemas, ensure_ascii=False, indent=2),
            "",
            "Phase hard rules:",
            *_phase_hard_rules(phase),
            "",
            "Common anti-patterns to avoid:",
            *_phase_anti_patterns(phase),
            "",
            "Task-family guidance:",
            *task_guidance,
            "",
            "Retry context:",
            *_retry_context_lines(retry_context),
            "",
            "Internal self-check before output:",
            *_self_check_lines(),
            "",
            "Plan style:",
            "- Keep parameters realistic and safe.",
            "- Parameter names must match schema exactly (no aliases in the output).",
            "- Use only declared fields with correct value types.",
            "- Include all required fields for each chosen tool.",
            "- Prefer concise plans (1-5 steps for actionable phases).",
            "- Return an empty steps list only if there is truly no safe actionable step for this phase.",
            "",
            "Return JSON with this exact shape:",
            json.dumps(
                {
                    "plan_name": f"{phase}_tool_plan",
                    "steps": [
                        {
                            "tool": "tool_name_from_allowlist",
                            "parameters": {"key": "value"},
                            "reasoning": "why this step is needed",
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )


__all__ = [
    "PHASE_TOOLLIST_SYSTEM_PROMPT",
    "build_phase_tolist_prompt",
]
