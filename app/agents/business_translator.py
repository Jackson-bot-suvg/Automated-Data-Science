"""Business Action Translator (BAT).

This node converts technical artifacts and model outputs into stakeholder-friendly,
action-oriented recommendations. It prefers deterministic fallbacks while using
LLM refinement when available.
"""

from __future__ import annotations

import asyncio
import contextvars
import csv
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import sys
import textwrap
from typing import Any
from queue import Queue
import threading

from app.llm.helpers import complete_json
from app.observability.langsmith import with_tracing_context
from app.observability.token_usage import record_token_usage, snapshot_token_usage
from app.prompts.business_prompt import (
    BUSINESS_TRANSLATOR_SYSTEM_PROMPT,
    build_business_translation_prompt,
)
from app.schemas.state_schema import GraphState
from app.tools import artifact_tools, file_tools

# ===========================================================================
# region — V3 Errors (active under _v3_ prefix until Task 14 swap)
# ===========================================================================


class _V3BusinessTranslatorError(Exception):
    """Raised internally inside BAT v3; never propagates past the public entrypoint."""


# ===========================================================================
# region — V3 Abort Delta
# ===========================================================================


def _v3_build_abort_delta(*, reason: str, error_class: str) -> dict:
    """Return the state delta the agent emits on any failure path.

    Shape is fixed so the node adapter sees the same keys regardless of failure mode.
    """
    return {
        "report_state": {
            "business_report_path": "",
            "limitations": [f"Business report aborted: {reason}"],
        },
        "artifacts": {
            "business_report_html": None,
            "business_translator": {
                "status": "aborted",
                "error": reason,
                "error_class": error_class,
                "charts_count": 0,
                "industry_label": None,
                "format": "html_v3",
            },
        },
    }


logger = logging.getLogger(__name__)


def run_business_translation(state: GraphState) -> dict:
    """Generate the business-facing HTML report (BAT v3).

    This function never raises. On any failure it returns an abort delta
    (see :func:`_v3_build_abort_delta`); on success it returns a success delta
    (see :func:`_v3_build_success_delta`).
    """
    try:
        return _v3_run_business_translation_impl(state)
    except _V3BusinessTranslatorError as exc:
        return _v3_build_abort_delta(reason=str(exc), error_class=type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 — last line of defence
        return _v3_build_abort_delta(reason=f"Unexpected error: {exc}", error_class=type(exc).__name__)


def _safe_read_text(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return ""
    try:
        return Path(p).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def _write_text(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


@dataclass(frozen=True)
class BusinessTranslation:
    executive_summary: str
    recommended_actions: list[str]
    report_markdown: str


@dataclass(frozen=True)
class HypothesisPlan:
    high_level_questions: list[str]
    drill_down_questions: list[str]
    evidence_plan: list[str]


@dataclass(frozen=True)
class InsightBrief:
    headline: str
    analysis_process: str
    evidence_used: list[str]
    business_implication: str


@dataclass(frozen=True)
class ActionBrief:
    decision: str
    analysis_process: str
    linked_insight_headline: str
    execution_hint: str


def _translate_to_business_actions(
    *,
    technical_report_text: str,
    validation_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    feature_importance: dict[str, Any],
    business_context: str,
    task_family: str,
    limitations: list[str],
    dataset_name: str,
    chart_paths: dict[str, str],
    artifact_evidence: dict[str, Any],
    domain_context: dict[str, Any],
) -> BusinessTranslation:
    """Deterministic BAT aligned with referenced literature themes.

    - Insight generation framing: capture patterns + why they matter + confidence signals
    - Multi-agent collaboration framing: produce structured outputs to hand-off downstream
    - Business decision framing: highlight decision-ready recommendations with risks/assumptions
    """

    metrics_lines = _format_metrics(validation_metrics)
    baseline_lines = _format_metrics(baseline_metrics)
    top_drivers = _top_k_features(feature_importance, k=4)

    executive_summary_parts: list[str] = []
    executive_opening = str(domain_context.get("executive_opening", "")).strip()
    if executive_opening:
        executive_summary_parts.append(executive_opening)
    if business_context:
        executive_summary_parts.append(f"Business context: {business_context}.")
    if task_family:
        executive_summary_parts.append(f"Task type: {task_family}.")
    metrics_snapshot = _metric_snapshot(validation_metrics)
    if metrics_snapshot:
        executive_summary_parts.append(f"Model performance snapshot: {metrics_snapshot}.")
    elif metrics_lines:
        executive_summary_parts.append(f"Model performance snapshot: {', '.join(metrics_lines)}.")
    if baseline_lines:
        executive_summary_parts.append(f"Baseline snapshot: {', '.join(baseline_lines)}.")
    if top_drivers:
        executive_summary_parts.append(f"Key drivers: {', '.join([d.name for d in top_drivers])}.")

    executive_summary = " ".join(executive_summary_parts).strip()
    if not executive_summary:
        executive_summary = "Business translation generated from technical outputs."

    recommended_actions = _recommend_actions(
        business_context=business_context,
        task_family=task_family,
        top_drivers=top_drivers,
        metrics=validation_metrics,
        limitations=limitations,
        domain_context=domain_context,
    )

    hypothesis_context = {
        "dataset_name": dataset_name,
        "task_family": task_family,
        "business_context": business_context,
        "validation_metrics": validation_metrics,
        "baseline_metrics": baseline_metrics,
        "top_drivers": [d.name for d in top_drivers],
        "limitations": limitations,
        "artifact_evidence": artifact_evidence,
        "domain_context": domain_context,
    }
    hypothesis_plan = _generate_hypothesis_plan(hypothesis_context)
    default_insight_points = _default_insight_points(
        task_family=task_family,
        validation_metrics=validation_metrics,
        baseline_metrics=baseline_metrics,
        top_drivers=top_drivers,
        hypothesis_plan=hypothesis_plan,
        artifact_evidence=artifact_evidence,
        domain_context=domain_context,
    )
    default_execution_suggestions = _execution_suggestions(
        task_family=task_family,
        business_context=business_context,
        top_drivers=top_drivers,
    )
    default_chart_guidance = _default_chart_guidance(chart_paths)
    default_risks = (
        limitations if limitations else ["No explicit limitations were provided by upstream nodes."]
    )
    default_insight_briefs = _build_default_insight_briefs(
        insight_points=default_insight_points,
        hypothesis_plan=hypothesis_plan,
        validation_metrics=validation_metrics,
        baseline_metrics=baseline_metrics,
        artifact_evidence=artifact_evidence,
        domain_context=domain_context,
    )
    default_action_briefs = _build_default_action_briefs(
        recommended_actions=recommended_actions,
        insight_briefs=default_insight_briefs,
        execution_suggestions=default_execution_suggestions,
        domain_context=domain_context,
    )

    report_md_fallback = _render_business_report_markdown(
        executive_summary=executive_summary,
        business_context=business_context,
        task_family=task_family,
        dataset_name=dataset_name,
        validation_metrics=validation_metrics,
        baseline_metrics=baseline_metrics,
        top_drivers=top_drivers,
        limitations=limitations,
        recommended_actions=recommended_actions,
        execution_suggestions=default_execution_suggestions,
        insight_points=default_insight_points,
        chart_guidance=default_chart_guidance,
        chart_paths=chart_paths,
        technical_report_excerpt=_excerpt(technical_report_text, max_chars=500),
        artifact_evidence=artifact_evidence,
        insight_briefs=default_insight_briefs,
        action_briefs=default_action_briefs,
        domain_context=domain_context,
    )

    fallback_payload = {
        "executive_summary": executive_summary,
        "industry_name": str(domain_context.get("industry_name", "General Predictive Operations")),
        "industry_confidence": 0.6,
        "industry_evidence": [],
        "scenario_definition": business_context
        or "Model-informed prioritization for operational decisions under constrained resources.",
        "operating_rules": [
            "Prioritize high-impact segments when resources are constrained.",
            "Tie intervention intensity to predicted risk or expected value.",
        ],
        "decision_economics": [
            "Cost lever: allocate scarce resources to high-priority cases first.",
            "Value lever: increase outcome quality through targeted interventions.",
            "Risk lever: monitor drift and enforce rollback thresholds.",
        ],
        "feature_to_business_mapping": [],
        "insight_briefs": _insight_briefs_to_payload(default_insight_briefs),
        "action_briefs": _action_briefs_to_payload(default_action_briefs),
        "insight_points": default_insight_points,
        "recommended_actions": recommended_actions,
        "execution_suggestions": default_execution_suggestions,
        "risks_and_limitations": default_risks,
        "stakeholder_explanation": executive_summary,
        "chart_guidance": default_chart_guidance,
        "hypothesis_summary": {
            "high_level_questions": hypothesis_plan.high_level_questions,
            "drill_down_questions": hypothesis_plan.drill_down_questions,
            "evidence_plan": hypothesis_plan.evidence_plan,
        },
        "report_markdown": report_md_fallback,
    }

    llm_context = {
        "single_call_mode": True,
        "report_language": "English",
        "report_style": "Low technical exposure, business narrative first",
        "anti_generic_requirement": "Use specific industry positioning and concrete operating logic; avoid generic labels.",
        "dataset_name": dataset_name,
        "task_family": task_family,
        "business_context": business_context,
        "validation_metrics": validation_metrics,
        "baseline_metrics": baseline_metrics,
        "feature_importance_summary": _compact_feature_importance_for_llm(feature_importance, k=10),
        "limitations": limitations,
        "artifact_evidence": _compact_artifact_evidence_for_llm(artifact_evidence),
        "domain_context": domain_context,
        "technical_report_excerpt": _excerpt(technical_report_text, max_chars=450),
        "chart_paths": chart_paths,
        "industry_name": domain_context.get("industry_name", ""),
        "business_signal_abstraction": _build_business_signal_abstraction(
            task_family=task_family,
            feature_importance=feature_importance,
            artifact_evidence=artifact_evidence,
            business_context=business_context,
        ),
        "hypothesis_plan": {
            "high_level_questions": hypothesis_plan.high_level_questions,
            "drill_down_questions": hypothesis_plan.drill_down_questions,
            "evidence_plan": hypothesis_plan.evidence_plan,
        },
    }
    llm_payload = _complete_json_with_timeout(
        system_prompt=BUSINESS_TRANSLATOR_SYSTEM_PROMPT,
        user_prompt=build_business_translation_prompt(context=llm_context),
        fallback=fallback_payload,
    )
    if not isinstance(llm_payload, dict):
        llm_payload = fallback_payload

    executive_summary = _coerce_string(
        llm_payload.get("executive_summary"),
        default=executive_summary,
    )
    recommended_actions = _coerce_str_list(
        llm_payload.get("recommended_actions"),
        default=recommended_actions,
    )
    insight_points = _coerce_str_list(
        llm_payload.get("insight_points"),
        default=fallback_payload["insight_points"],
    )
    execution_suggestions = _coerce_str_list(
        llm_payload.get("execution_suggestions"),
        default=fallback_payload["execution_suggestions"],
    )
    risks_and_limitations = _coerce_str_list(
        llm_payload.get("risks_and_limitations"),
        default=fallback_payload["risks_and_limitations"],
    )
    chart_guidance = _coerce_str_list(
        llm_payload.get("chart_guidance"),
        default=fallback_payload["chart_guidance"],
    )
    insight_briefs = _coerce_insight_briefs(
        llm_payload.get("insight_briefs"),
        default=default_insight_briefs,
    )
    action_briefs = _coerce_action_briefs(
        llm_payload.get("action_briefs"),
        default=default_action_briefs,
    )
    llm_industry_name = _coerce_string(
        llm_payload.get("industry_name"),
        default=str(domain_context.get("industry_name", "")),
    )
    if llm_industry_name:
        domain_context["industry_name"] = llm_industry_name
    llm_scenario_definition = _coerce_string(
        llm_payload.get("scenario_definition"),
        default=str(domain_context.get("scenario_definition", "")),
    )
    if llm_scenario_definition:
        domain_context["scenario_definition"] = llm_scenario_definition
    domain_context["operating_rules"] = _coerce_str_list(
        llm_payload.get("operating_rules"),
        default=_coerce_str_list(domain_context.get("operating_rules"), default=[]),
    )
    domain_context["decision_economics"] = _coerce_str_list(
        llm_payload.get("decision_economics"),
        default=_coerce_str_list(domain_context.get("decision_economics"), default=[]),
    )
    raw_feature_mapping = llm_payload.get("feature_to_business_mapping")
    if isinstance(raw_feature_mapping, list):
        mapping_lines: list[str] = []
        for item in raw_feature_mapping[:5]:
            if not isinstance(item, dict):
                continue
            feature = str(item.get("feature", "")).strip()
            meaning = str(item.get("business_meaning", "")).strip()
            lever = str(item.get("decision_lever", "")).strip()
            if feature and (meaning or lever):
                mapping_lines.append(
                    f"`{feature}` -> {meaning or 'meaning not specified'}; decision lever: {lever or 'not specified'}."
                )
        if mapping_lines:
            domain_context["feature_business_mapping_lines"] = mapping_lines
    report_md = _coerce_string(llm_payload.get("report_markdown"), default="")
    if not _has_required_report_sections(report_md):
        report_md = _render_business_report_markdown(
            executive_summary=executive_summary,
            business_context=business_context,
            task_family=task_family,
            dataset_name=dataset_name,
            validation_metrics=validation_metrics,
            baseline_metrics=baseline_metrics,
            top_drivers=top_drivers,
            limitations=risks_and_limitations,
            recommended_actions=recommended_actions,
            execution_suggestions=execution_suggestions,
            insight_points=insight_points,
            chart_guidance=chart_guidance,
            chart_paths=chart_paths,
            technical_report_excerpt=_excerpt(technical_report_text, max_chars=500),
            artifact_evidence=artifact_evidence,
            insight_briefs=insight_briefs,
            action_briefs=action_briefs,
            domain_context=domain_context,
        )

    # One-call mode: critic reflection is intentionally disabled.
    if not _has_required_report_sections(report_md):
        report_md = _render_business_report_markdown(
            executive_summary=executive_summary,
            business_context=business_context,
            task_family=task_family,
            dataset_name=dataset_name,
            validation_metrics=validation_metrics,
            baseline_metrics=baseline_metrics,
            top_drivers=top_drivers,
            limitations=risks_and_limitations,
            recommended_actions=recommended_actions,
            execution_suggestions=execution_suggestions,
            insight_points=insight_points,
            chart_guidance=chart_guidance,
            chart_paths=chart_paths,
            technical_report_excerpt=_excerpt(technical_report_text, max_chars=500),
            artifact_evidence=artifact_evidence,
            insight_briefs=insight_briefs,
            action_briefs=action_briefs,
            domain_context=domain_context,
        )
    report_md = _polish_report_text(report_md)

    return BusinessTranslation(
        executive_summary=executive_summary,
        recommended_actions=recommended_actions,
        report_markdown=report_md,
    )


@dataclass(frozen=True)
class FeatureDriver:
    name: str
    importance: float


def _top_k_features(feature_importance: dict[str, Any], *, k: int) -> list[FeatureDriver]:
    drivers: list[FeatureDriver] = []
    for name, raw in feature_importance.items():
        try:
            importance = float(raw)
        except (TypeError, ValueError):
            continue
        drivers.append(FeatureDriver(name=str(name), importance=importance))
    drivers.sort(key=lambda d: d.importance, reverse=True)
    return drivers[: max(0, k)]


def _format_metrics(metrics: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in sorted(metrics.keys()):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            parts.append(f"{key}={val:.4g}")
        elif isinstance(val, dict):
            compact = _compact_model_metric_line(str(key), val)
            if compact:
                parts.append(compact)
        else:
            parts.append(f"{key}={val}")
    return parts


def _compact_model_metric_line(model_name: str, values: dict[str, Any]) -> str:
    highlights: list[str] = []
    for metric_key, label in (
        ("cv_accuracy_mean", "acc"),
        ("cv_f1_mean", "f1"),
        ("cv_auc_mean", "auc"),
        ("rmse", "rmse"),
        ("mae", "mae"),
        ("r2", "r2"),
    ):
        raw = values.get(metric_key)
        if isinstance(raw, (int, float)):
            highlights.append(f"{label}={raw:.4g}")
    if highlights:
        return f"{model_name}({', '.join(highlights)})"
    return ""


def _metric_snapshot(metrics: dict[str, Any]) -> str:
    best_model = ""
    best_acc = None
    for model_name, model_metrics in metrics.items():
        if not isinstance(model_metrics, dict):
            continue
        score = model_metrics.get("cv_accuracy_mean")
        if isinstance(score, (int, float)) and (best_acc is None or score > best_acc):
            best_acc = float(score)
            best_model = str(model_name)
    if best_model and best_acc is not None:
        return f"best observed cross-validated accuracy is {best_acc:.4g} ({best_model})"

    flat_metrics = _format_metrics(metrics)
    if flat_metrics:
        return ", ".join(flat_metrics[:3])
    return ""


def _recommend_actions(
    *,
    business_context: str,
    task_family: str,
    top_drivers: list[FeatureDriver],
    metrics: dict[str, Any],
    limitations: list[str],
    domain_context: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    context_actions = domain_context.get("action_hints")
    if isinstance(context_actions, list):
        actions.extend(str(item).strip() for item in context_actions if str(item).strip())

    if top_drivers:
        primary = top_drivers[0].name
        actions.append(f"P0: Review and operationalize monitoring for the top driver `{primary}` (define thresholds, owners, and escalation playbooks).")
        if len(top_drivers) >= 2:
            secondary = top_drivers[1].name
            actions.append(f"P1: Design interventions or policy levers for `{secondary}` and validate impact via A/B test or retrospective analysis.")
        actions.append("P1: Create a stakeholder-facing data dictionary for the key drivers and confirm they align with business controllability (actionable vs. observational).")

    if task_family in {"classification", "binary_classification", "multiclass_classification"}:
        actions.append("P0: Agree on the operating point (threshold) with stakeholders and document trade-offs (false positives vs. false negatives).")
    if task_family == "regression":
        actions.append("P0: Define acceptable error bands and decision rules (e.g., actions triggered when predicted value crosses a business threshold).")

    if metrics:
        actions.append("P1: Set up weekly model performance reporting (core metric + drift checks) and define retraining triggers.")

    if limitations:
        actions.append("P1: Track limitations as risks with mitigation owners; prioritize collecting missing signals or improving label quality where applicable.")

    if business_context:
        actions.append(f"P2: Tailor the narrative to the business context: `{business_context}` (who decides, what changes, and expected ROI).")

    actions.extend(_industry_specific_action_hints(domain_context))

    if not actions:
        actions.append("P0: Validate that technical artifacts are complete (technical report, metrics, feature importance) before drafting business recommendations.")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in actions:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _render_business_report_markdown(
    *,
    executive_summary: str,
    business_context: str,
    task_family: str,
    dataset_name: str,
    validation_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    top_drivers: list[FeatureDriver],
    limitations: list[str],
    recommended_actions: list[str],
    execution_suggestions: list[str],
    insight_points: list[str],
    chart_guidance: list[str],
    chart_paths: dict[str, str],
    technical_report_excerpt: str,
    artifact_evidence: dict[str, Any],
    insight_briefs: list[InsightBrief],
    action_briefs: list[ActionBrief],
    domain_context: dict[str, Any],
) -> str:
    industry_block = str(domain_context.get("industry_name") or "General Predictive Operations")
    context_block = (
        str(domain_context.get("scenario_definition", "")).strip()
        or business_context
        or "Model-informed decision support for core business operations."
    )
    environment_scan = domain_context.get("environment_scan")
    environment_lines = (
        [f"- {str(item)}" for item in environment_scan if str(item).strip()]
        if isinstance(environment_scan, list) and environment_scan
        else ["- Focus on decision quality, operational risk, and execution feasibility."]
    )
    operating_rules = domain_context.get("operating_rules")
    operating_rule_lines = (
        [f"- {str(item)}" for item in operating_rules if str(item).strip()]
        if isinstance(operating_rules, list) and operating_rules
        else [
            "- Prioritize high-impact segments when response resources are constrained.",
            "- Tie intervention intensity to predicted severity or expected value.",
        ]
    )
    decision_economics = domain_context.get("decision_economics")
    economics_lines = (
        [f"- {str(item)}" for item in decision_economics if str(item).strip()]
        if isinstance(decision_economics, list) and decision_economics
        else [
            "- Cost lever: reduce wasted intervention capacity through better prioritization.",
            "- Value lever: improve outcome quality using segment-specific operating policies.",
            "- Risk lever: enforce drift monitoring and rollback thresholds.",
        ]
    )
    feature_mapping_lines = domain_context.get("feature_business_mapping_lines")
    if isinstance(feature_mapping_lines, list) and feature_mapping_lines:
        mapping_lines = [f"- {str(item)}" for item in feature_mapping_lines if str(item).strip()]
    elif top_drivers:
        mapping_lines = [
            f"- `{driver.name}` acts as a decision signal for prioritization and differentiated response."
            for driver in top_drivers[:4]
        ]
    else:
        mapping_lines = ["- Feature-level business mapping is currently limited by available signal metadata."]
    insight_lines = [f"- {line}" for line in insight_points[:5]] or ["- Evidence-backed insights are currently limited; proceed with staged validation."]
    execution_lines = [f"- {line}" for line in execution_suggestions[:4]] or ["- Weekly governance cadence with explicit KPI and risk triggers."]
    limitations_lines = [f"- {line}" for line in limitations[:4]] or ["- Key limitations are not fully specified; validate assumptions in pilot rollout."]
    metrics_lines = _format_metrics(validation_metrics)
    baseline_lines = _format_metrics(baseline_metrics)
    core_metrics_lines: list[str] = []
    if metrics_lines:
        core_metrics_lines.append(f"- Validation performance snapshot: {', '.join(metrics_lines[:4])}.")
    if baseline_lines:
        core_metrics_lines.append(f"- Baseline reference snapshot: {', '.join(baseline_lines[:4])}.")
    if not core_metrics_lines:
        core_metrics_lines.append("- Core metric evidence is currently limited; use pilot KPIs for validation.")

    tool_method_lines: list[str] = []
    if "classification" in task_family:
        tool_method_lines.append("- Use a classification algorithm to convert feature signals into tiered decision outputs.")
    elif task_family == "regression":
        tool_method_lines.append("- Use a regression algorithm to forecast outcome magnitude and prioritize interventions by expected value.")
    else:
        tool_method_lines.append("- Use predictive modeling to rank opportunities and risks for operational prioritization.")
    if mapping_lines:
        tool_method_lines.append("- Translate top signals into action levers through feature-to-business mapping.")

    action_card_lines = _render_action_cards(
        action_briefs=action_briefs,
        recommended_actions=recommended_actions,
        execution_suggestions=execution_suggestions,
    )
    summary_lines = _business_brief_summary(
        executive_summary=executive_summary,
        recommended_actions=recommended_actions,
    )
    return "\n".join(
        [
            "# Business Report",
            "",
            "## Preface",
            executive_summary.strip() or "This report translates technical evidence into decision-ready business guidance.",
            "",
            "## Business Context Modeling",
            "### Scenario Definition",
            context_block,
            "",
            "### Industry Positioning",
            f"- {industry_block}",
            "",
            "### Operating Rules",
            *operating_rule_lines,
            "",
            "### Core Metrics",
            *core_metrics_lines,
            "",
            "### Tooling Method",
            *tool_method_lines,
            "",
            "### Business Environment Scan",
            *environment_lines,
            "",
            "### Decision Economics",
            *economics_lines,
            "",
            "## Core Insight Discovery",
            *insight_lines,
            "",
            "### Feature-to-Decision Mapping",
            *mapping_lines,
            "",
            "## Strategic Action Recommendations",
            *action_card_lines,
            "",
            "### Execution Cadence And Governance",
            *execution_lines,
            "",
            "### Risks And Limitations",
            *limitations_lines,
            "",
            "## Business Brief Summary",
            *summary_lines,
            "",
        ]
    )


def _business_brief_summary(
    *, executive_summary: str, recommended_actions: list[str]
) -> list[str]:
    first_action = (
        recommended_actions[0]
        if recommended_actions
        else "Define owners, cadence, and KPI checkpoints before rollout."
    )
    lines = [
        "This report reframes technical outcomes into an executable business operating model.",
        f"First execution lever: {first_action}",
    ]
    if executive_summary.strip():
        lines.append(f"Leadership takeaway: {executive_summary.strip()}")
    return lines


def _render_action_cards(
    *,
    action_briefs: list[ActionBrief],
    recommended_actions: list[str],
    execution_suggestions: list[str],
) -> list[str]:
    lines: list[str] = []
    if action_briefs:
        for idx, brief in enumerate(action_briefs[:5], start=1):
            lines.extend(
                [
                    f"{idx}. **Strategic Theme:** {brief.linked_insight_headline}",
                    f"   - **Recommendation:** {brief.decision}",
                    f"   - **Business Objective:** {brief.analysis_process}",
                    f"   - **Execution Blueprint:** {brief.execution_hint}",
                    "",
                ]
            )
    else:
        for idx, action in enumerate(recommended_actions[:5], start=1):
            exec_hint = execution_suggestions[min(idx - 1, len(execution_suggestions) - 1)] if execution_suggestions else "Owner/cadence/KPI/risk to be confirmed in rollout planning."
            lines.extend(
                [
                    f"{idx}. **Strategic Theme:** Prioritized action package {idx}",
                    f"   - **Recommendation:** {action}",
                    "   - **Business Objective:** Improve operating efficiency, decision quality, and controlled risk reduction.",
                    f"   - **Execution Blueprint:** {exec_hint}",
                    "",
                ]
            )
    if not lines:
        return ["1. **Strategic Theme:** Initial rollout", "   - **Recommendation:** Define pilot scope and action owners.", "   - **Business Objective:** Validate measurable business impact before scale-up.", "   - **Execution Blueprint:** Weekly cadence, KPI checkpoint, and explicit risk threshold."]
    return lines


def _has_required_report_sections(report_markdown: str) -> bool:
    text = (report_markdown or "").strip()
    required = [
        "## Preface",
        "## Business Context Modeling",
        "## Core Insight Discovery",
        "## Strategic Action Recommendations",
        "## Business Brief Summary",
        "### Industry Positioning",
        "### Operating Rules",
        "### Core Metrics",
        "### Tooling Method",
        "Strategic Theme",
        "Recommendation",
        "Business Objective",
        "Execution Blueprint",
    ]
    return all(section in text for section in required)


def _what_model_can_support(
    *,
    task_family: str,
    validation_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    if task_family in {"classification", "binary_classification", "multiclass_classification"}:
        lines.append("- Supports risk ranking and threshold-based intervention decisions.")
    elif task_family == "regression":
        lines.append("- Supports numeric forecasting and threshold-triggered operational decisions.")
    else:
        lines.append("- Supports model-informed decision making, with domain review before rollout.")

    deltas = _metric_deltas(validation_metrics, baseline_metrics)
    if deltas:
        lines.append(f"- Validation vs baseline deltas: {', '.join(deltas)}.")
    elif validation_metrics:
        lines.append("- Validation metrics are available; baseline comparison is not available.")
    else:
        lines.append("- Validation metrics are not available yet.")

    return lines


def _metric_deltas(validation_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> list[str]:
    deltas: list[str] = []
    for key in sorted(set(validation_metrics.keys()) & set(baseline_metrics.keys())):
        val = validation_metrics.get(key)
        base = baseline_metrics.get(key)
        if isinstance(val, (int, float)) and isinstance(base, (int, float)):
            sign = "+" if (val - base) >= 0 else ""
            deltas.append(f"{key}={sign}{(val - base):.4g}")
    return deltas


def _default_insight_points(
    *,
    task_family: str,
    validation_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    top_drivers: list[FeatureDriver],
    hypothesis_plan: HypothesisPlan,
    artifact_evidence: dict[str, Any],
    domain_context: dict[str, Any],
) -> list[str]:
    insights: list[str] = []
    domain_insights = domain_context.get("insight_evidence_points")
    if isinstance(domain_insights, list):
        insights.extend(str(item).strip() for item in domain_insights if str(item).strip())
    deltas = _metric_deltas(validation_metrics, baseline_metrics)
    if deltas:
        insights.append(
            "Performance improved versus baseline on key metrics: "
            + ", ".join(deltas)
            + "."
        )
    elif validation_metrics:
        insights.append(
            "Validation metrics are available, but no baseline comparison was provided."
        )
    else:
        insights.append("Model validation metrics are currently unavailable.")

    if top_drivers:
        lead = top_drivers[0]
        insights.append(
            f"The leading business driver is {lead.name}, indicating this factor has the strongest effect on predictions."
        )
    else:
        insights.append(
            "No feature-importance evidence is available, so driver-level interpretation is limited."
        )

    if task_family == "regression":
        insights.append(
            "The model supports value forecasting, which helps prioritize actions by expected magnitude."
        )
    else:
        insights.append(
            "The model supports risk segmentation, enabling targeted interventions for different user groups."
        )

    if hypothesis_plan.high_level_questions:
        insights.append(
            "Key business question addressed: "
            + hypothesis_plan.high_level_questions[0]
        )
    insights.extend(_artifact_business_insights(artifact_evidence))
    return insights


def _default_chart_guidance(chart_paths: dict[str, str]) -> list[str]:
    notes: list[str] = []
    if chart_paths.get("metrics_chart"):
        notes.append(
            "Metrics comparison chart highlights where the model outperforms or underperforms baseline."
        )
    if chart_paths.get("drivers_chart"):
        notes.append(
            "Key driver chart shows which factors matter most, helping prioritize business levers."
        )
    if not notes:
        notes.append("No charts were generated from available inputs.")
    return notes


def _artifact_business_insights(artifact_evidence: dict[str, Any]) -> list[str]:
    summaries = artifact_evidence.get("artifact_summaries")
    if not isinstance(summaries, dict) or not summaries:
        return []

    insights: list[str] = []
    for rel_path, summary in summaries.items():
        if not isinstance(summary, dict):
            continue
        item_type = str(summary.get("type", "")).lower()
        if item_type == "csv":
            rows = summary.get("rows")
            columns = summary.get("columns")
            if isinstance(rows, int) and isinstance(columns, list):
                preview_cols = ", ".join(str(c) for c in columns[:5]) or "unknown columns"
                insights.append(
                    f"Artifact evidence from `{rel_path}` shows {rows} records with core fields ({preview_cols}), supporting operational segmentation and monitoring design."
                )
                continue
        if item_type == "markdown":
            text = str(summary.get("summary", "")).strip()
            if text:
                insights.append(
                    f"Process documentation in `{rel_path}` provides pipeline context that improves business interpretability and rollout confidence."
                )
                continue
        if item_type == "json":
            keys = summary.get("keys")
            if isinstance(keys, list) and keys:
                insights.append(
                    f"Structured outputs in `{rel_path}` include key fields ({', '.join(str(k) for k in keys[:5])}), enabling traceable decision governance."
                )

    return insights[:3]


def _build_default_insight_briefs(
    *,
    insight_points: list[str],
    hypothesis_plan: HypothesisPlan,
    validation_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    artifact_evidence: dict[str, Any],
    domain_context: dict[str, Any],
) -> list[InsightBrief]:
    briefs: list[InsightBrief] = []
    deltas = _metric_deltas(validation_metrics, baseline_metrics)
    evidence_lines: list[str] = []
    if deltas:
        evidence_lines.append("Validation vs baseline metric delta analysis: " + ", ".join(deltas))
    artifact_paths = artifact_evidence.get("artifact_paths")
    if isinstance(artifact_paths, list) and artifact_paths:
        evidence_lines.append("Referenced artifacts: " + ", ".join(str(p) for p in artifact_paths[:5]))

    default_headlines = domain_context.get("insight_headlines")
    if not isinstance(default_headlines, list) or not default_headlines:
        default_headlines = [
            "Model performance is now decision-usable for this scenario.",
            "Operational segmentation is feasible with current signals.",
            "Current deployment confidence depends on governance controls.",
        ]
    for idx, headline in enumerate(default_headlines):
        point = insight_points[idx] if idx < len(insight_points) else "Insight synthesized from technical review."
        process = str(domain_context.get("insight_method", "")).strip() or (
            "Triangulated model metrics, baseline comparison, and artifact summaries from the technical report."
        )
        if idx == 1 and hypothesis_plan.drill_down_questions:
            process = (
                "Used drill-down questioning to connect feature-level and dataset-level evidence to business operations."
            )
        implication = point
        briefs.append(
            InsightBrief(
                headline=headline,
                analysis_process=process,
                evidence_used=evidence_lines or ["Technical report evidence was used where available."],
                business_implication=implication,
            )
        )
    return briefs


def _build_default_action_briefs(
    *,
    recommended_actions: list[str],
    insight_briefs: list[InsightBrief],
    execution_suggestions: list[str],
    domain_context: dict[str, Any],
) -> list[ActionBrief]:
    briefs: list[ActionBrief] = []
    insight_titles = [item.headline for item in insight_briefs]
    action_method = str(domain_context.get("action_method", "")).strip()
    for idx, action in enumerate(recommended_actions[:3]):
        linked = insight_titles[idx] if idx < len(insight_titles) else (insight_titles[0] if insight_titles else "Primary insight")
        hint = execution_suggestions[idx] if idx < len(execution_suggestions) else "Define owner, cadence, KPI, and risk trigger."
        briefs.append(
            ActionBrief(
                decision=action,
                analysis_process=action_method or (
                    "Mapped insight-to-decision by evaluating risk/benefit trade-offs and choosing the lowest-friction operational intervention."
                ),
                linked_insight_headline=linked,
                execution_hint=hint,
            )
        )
    return briefs


def _insight_briefs_to_payload(briefs: list[InsightBrief]) -> list[dict[str, Any]]:
    return [
        {
            "headline": b.headline,
            "analysis_process": b.analysis_process,
            "evidence_used": b.evidence_used,
            "business_implication": b.business_implication,
        }
        for b in briefs
    ]


def _action_briefs_to_payload(briefs: list[ActionBrief]) -> list[dict[str, Any]]:
    return [
        {
            "decision": b.decision,
            "analysis_process": b.analysis_process,
            "linked_insight_headline": b.linked_insight_headline,
            "execution_hint": b.execution_hint,
        }
        for b in briefs
    ]


def _coerce_insight_briefs(value: Any, *, default: list[InsightBrief]) -> list[InsightBrief]:
    if not isinstance(value, list):
        return list(default)
    parsed: list[InsightBrief] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline", "")).strip()
        analysis_process = str(item.get("analysis_process", "")).strip()
        evidence_raw = item.get("evidence_used")
        evidence_used = (
            [str(x).strip() for x in evidence_raw if str(x).strip()]
            if isinstance(evidence_raw, list)
            else []
        )
        implication = str(item.get("business_implication", "")).strip()
        if not headline:
            continue
        parsed.append(
            InsightBrief(
                headline=headline,
                analysis_process=analysis_process or "Analysis process not explicitly provided.",
                evidence_used=evidence_used or ["Evidence source not explicitly provided."],
                business_implication=implication or "Business implication not explicitly provided.",
            )
        )
    return parsed or list(default)


def _coerce_action_briefs(value: Any, *, default: list[ActionBrief]) -> list[ActionBrief]:
    if not isinstance(value, list):
        return list(default)
    parsed: list[ActionBrief] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision", "")).strip()
        analysis_process = str(item.get("analysis_process", "")).strip()
        linked = str(item.get("linked_insight_headline", "")).strip()
        execution_hint = str(item.get("execution_hint", "")).strip()
        if not decision:
            continue
        parsed.append(
            ActionBrief(
                decision=decision,
                analysis_process=analysis_process or "Decision rationale not explicitly provided.",
                linked_insight_headline=linked or "Primary insight",
                execution_hint=execution_hint or "Execution details to be finalized by business owner.",
            )
        )
    return parsed or list(default)


def _render_insight_briefs(briefs: list[InsightBrief]) -> list[str]:
    if not briefs:
        return ["- (not available)"]
    lines: list[str] = []
    for idx, brief in enumerate(briefs, start=1):
        lines.append(f"- Insight {idx} (Viewpoint): {brief.headline}")
        lines.append(f"  - Analysis process: {brief.analysis_process}")
        lines.append("  - Evidence used:")
        for evidence in brief.evidence_used:
            lines.append(f"    - {evidence}")
        lines.append(f"  - Business implication: {brief.business_implication}")
    return lines


def _render_action_briefs(briefs: list[ActionBrief]) -> list[str]:
    if not briefs:
        return ["- (not available)"]
    lines: list[str] = []
    for idx, brief in enumerate(briefs, start=1):
        lines.append(f"- Action {idx} (Decision): {brief.decision}")
        lines.append(f"  - Linked insight: {brief.linked_insight_headline}")
        lines.append(f"  - Analysis process: {brief.analysis_process}")
        lines.append(f"  - Execution hint: {brief.execution_hint}")
    return lines


def _generate_hypothesis_plan(context: dict[str, Any]) -> HypothesisPlan:
    dataset_name = str(context.get("dataset_name", "")).strip() or "this dataset"
    task_family = str(context.get("task_family", "")).strip() or "classification"
    top_drivers = context.get("top_drivers")
    top_driver = ""
    if isinstance(top_drivers, list) and top_drivers:
        top_driver = str(top_drivers[0]).strip()
    has_baseline = bool(context.get("baseline_metrics"))
    high_level = [
        f"Which near-term business decisions for {dataset_name} can be improved using the current {task_family} model outputs?"
    ]
    drill_down = [
        "Which metric signals are stable enough to support operational rollout?",
        "Where do model limitations create execution risk or decision blind spots?",
    ]
    if has_baseline:
        drill_down.insert(0, "Which metrics improved against baseline and are the gains material for business KPIs?")
    if top_driver:
        drill_down.append(f"How can teams influence the top driver `{top_driver}` through controllable interventions?")
    evidence = [
        "Use validation and baseline deltas to calibrate confidence and rollout scope.",
        "Use top feature drivers and artifact evidence to tie recommendations to operational levers.",
        "Explicitly map limitations to mitigations before assigning owners and cadence.",
    ]
    return HypothesisPlan(
        high_level_questions=high_level,
        drill_down_questions=drill_down,
        evidence_plan=evidence,
    )


def _apply_critic_reflection(
    *,
    report_markdown: str,
    llm_context: dict[str, Any],
    fallback_markdown: str,
) -> str:
    critic_fallback = {
        "needs_revision": False,
        "major_issues": [],
        "revision_instructions": [],
        "quality_score_0_to_10": 7.0,
    }
    critic_payload = _complete_json_with_timeout(
        system_prompt=BUSINESS_TRANSLATOR_CRITIC_SYSTEM_PROMPT,
        user_prompt=build_business_critic_prompt(
            report_markdown=report_markdown,
            context=llm_context,
        ),
        fallback=critic_fallback,
    )
    if not isinstance(critic_payload, dict):
        return report_markdown
    needs_revision = bool(critic_payload.get("needs_revision"))
    instructions = _coerce_str_list(
        critic_payload.get("revision_instructions"),
        default=[],
    )
    if not needs_revision or not instructions:
        return report_markdown

    revision_fallback = {"report_markdown": report_markdown}
    revision_payload = _complete_json_with_timeout(
        system_prompt=BUSINESS_TRANSLATOR_SYSTEM_PROMPT,
        user_prompt=(
            "Revise the report using these critic instructions:\n"
            + "\n".join(f"- {item}" for item in instructions)
            + "\n\nCurrent report:\n"
            + report_markdown
        ),
        fallback=revision_fallback,
    )
    if not isinstance(revision_payload, dict):
        return report_markdown
    revised = _coerce_string(
        revision_payload.get("report_markdown"),
        default=report_markdown,
    )
    return revised or fallback_markdown


def _should_enable_critic() -> bool:
    return os.getenv("AGENT_BUSINESS_TRANSLATOR_ENABLE_CRITIC", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _should_generate_charts() -> bool:
    value = os.getenv("AGENT_BUSINESS_TRANSLATOR_GENERATE_CHARTS", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _polish_report_text(report_markdown: str) -> str:
    text = (report_markdown or "").strip()
    if not text:
        return report_markdown

    replacements = {
        "Validation metrics are available, but no baseline comparison was provided.": (
            "Validation metrics are available; baseline comparison is not yet available, so decisions should be phased."
        ),
        "Model validation metrics are currently unavailable.": (
            "Model validation metrics are currently unavailable, so recommendations should be treated as directional."
        ),
        "No feature-importance evidence is available, so driver-level interpretation is limited.": (
            "Feature-importance evidence is currently limited, so driver-level interpretation should be treated with caution."
        ),
        "Validation vs baseline deltas:": "Compared with baseline, key metric shifts are:",
        "Supports model-informed decision making, with domain review before rollout.": (
            "Supports decision-making with model evidence, but domain review is required before rollout."
        ),
        "Validation metrics are available; baseline comparison is not available.": (
            "Validation metrics are available, while baseline comparison is not yet available."
        ),
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _industry_specific_action_hints(domain_context: dict[str, Any]) -> list[str]:
    industry = str(domain_context.get("industry_name", "")).lower()
    hints: list[str] = []
    if "retention" in industry or "subscription" in industry:
        hints.append(
            "P0: Prioritize interventions by expected retention value and cap treatment cost per cohort."
        )
    elif "financial" in industry or "credit" in industry or "risk" in industry:
        hints.append(
            "P0: Formalize threshold governance with weekly loss/approval trade-off review."
        )
    elif "healthcare" in industry or "population" in industry:
        hints.append(
            "P0: Stage risk-tier interventions with clear clinical review and capacity constraints."
        )
    elif "manufacturing" in industry or "quality" in industry:
        hints.append(
            "P0: Focus quality controls on the highest-risk process windows and monitor false-alarm cost."
        )
    elif "education" in industry:
        hints.append(
            "P0: Deploy early-support interventions by risk band and track outcome uplift per term."
        )
    elif "commercial" in industry or "pricing" in industry:
        hints.append(
            "P0: Tie forecast-driven decisions to explicit margin and service-level guardrails."
        )
    elif "transport" in industry or "travel" in industry or "tourism" in industry:
        hints.append(
            "P0: Prioritize safety-response SOP upgrades in high-risk passenger segments and high-density cabin/service zones."
        )
    else:
        hints.append(
            "P2: Treat current results as decision support; validate impact with staged rollout and KPI monitoring."
        )
    return hints


def _build_business_domain_context(
    *,
    dataset_name: str,
    train_path: str,
    business_context: str,
    technical_report_text: str,
    artifact_evidence: dict[str, Any],
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "industry_name": "",
        "executive_opening": "",
        "scenario_definition": "",
        "environment_scan": [],
        "operating_rules": [],
        "decision_economics": [],
        "insight_headlines": [],
        "insight_evidence_points": [],
        "action_hints": [],
        "insight_method": "",
        "action_method": "",
    }
    heuristic_profile = _infer_industry_profile(
        dataset_name=dataset_name,
        business_context=business_context,
        technical_report_text=technical_report_text,
        artifact_evidence=artifact_evidence,
    )
    profile = heuristic_profile
    if profile:
        context.update(
            {
                "industry_name": profile["industry_name"],
                "executive_opening": profile["executive_opening"],
                "scenario_definition": profile.get("scenario_definition", ""),
                "environment_scan": profile["environment_scan"],
                "operating_rules": profile.get("operating_rules", []),
                "decision_economics": profile.get("decision_economics", []),
                "insight_headlines": profile["insight_headlines"],
                "action_hints": profile["action_hints"],
                "insight_method": profile["insight_method"],
                "action_method": profile["action_method"],
            }
        )
    return context


def _infer_industry_profile(
    *,
    dataset_name: str,
    business_context: str,
    technical_report_text: str,
    artifact_evidence: dict[str, Any],
) -> dict[str, Any]:
    column_signal = _artifact_column_signal(artifact_evidence)
    blob = " ".join(
        [
            dataset_name.lower(),
            business_context.lower(),
            _excerpt(technical_report_text.lower(), max_chars=800),
            _artifact_text_signal(artifact_evidence),
            " ".join(column_signal),
        ]
    )
    blob_norm = blob.lower()
    profiles: list[dict[str, Any]] = [
        {
            "industry_name": "Telecom / Subscription Retention",
            "keywords": ["churn", "retention", "subscription", "customer lifetime", "cancel"],
            "executive_opening": "This analysis is positioned for subscription growth teams balancing retention ROI, service quality, and intervention cost.",
            "environment_scan": [
                "Industry framing: recurring-revenue business with high sensitivity to churn and reactivation spend.",
                "Core KPIs: churn rate, retention uplift, intervention conversion, and campaign unit economics.",
                "Operational rule: prioritize interventions where expected retention value clearly exceeds treatment cost.",
            ],
            "insight_headlines": [
                "Risk segmentation can improve retention efficiency.",
                "Driver-level signals can guide lower-cost interventions.",
                "Rollout confidence depends on threshold governance.",
            ],
            "action_hints": [
                "P0: Deploy tiered save-offer policies by churn-risk segment and track incremental retention lift.",
                "P1: Align outreach channel and timing to top churn drivers for each cohort.",
                "P1: Add holdout-based ROI tracking to prevent over-incentivizing low-risk customers.",
            ],
            "insight_method": "Combined model performance, cohort segmentation evidence, and driver signals to identify retention levers with measurable payoff.",
            "action_method": "Converted insight priorities into retention playbooks with owner, cadence, budget guardrails, and uplift KPI checks.",
        },
        {
            "industry_name": "E-commerce / Digital Retail",
            "keywords": ["order", "basket", "cart", "sku", "conversion", "gmv", "aov", "session"],
            "executive_opening": "This analysis supports growth and operations teams balancing conversion efficiency, basket economics, and customer experience.",
            "environment_scan": [
                "Industry framing: high-velocity digital commerce with margin pressure and demand variability.",
                "Core KPIs: conversion rate, average order value, repeat purchase rate, and fulfillment cost.",
                "Operational rule: prioritize interventions that improve conversion or margin without harming customer trust.",
            ],
            "insight_headlines": [
                "Customer and session signals can sharpen conversion targeting.",
                "Product and pricing drivers can improve basket economics.",
                "Operational rollout should be staged by commercial risk.",
            ],
            "action_hints": [
                "P0: Launch cohort-level conversion interventions and track uplift by channel and segment.",
                "P1: Use driver insights to optimize merchandising and discount policy with guardrails on margin.",
                "P1: Build weekly decision dashboards for conversion, AOV, and return-risk drift.",
            ],
            "insight_method": "Connected model evidence to customer journey stages and commercial KPI trade-offs.",
            "action_method": "Mapped each insight to executable levers in pricing, merchandising, and retention operations.",
        },
        {
            "industry_name": "Financial Risk / Credit",
            "keywords": ["credit", "loan", "default", "fraud", "risk score", "bank", "balance", "account"],
            "executive_opening": "This analysis supports risk-policy decisions where growth targets must be balanced against loss exposure and compliance obligations.",
            "environment_scan": [
                "Industry framing: regulated decisioning environment with strict explainability and fairness expectations.",
                "Core KPIs: approval quality, default/loss rate, fraud capture, and manual review efficiency.",
                "Operational rule: tune risk thresholds with explicit trade-off between approval volume and loss cost.",
            ],
            "insight_headlines": [
                "Risk ranking quality can improve portfolio outcomes.",
                "Decision threshold policy drives business-loss trade-offs.",
                "Governance quality determines safe model adoption.",
            ],
            "action_hints": [
                "P0: Define segment-level decision thresholds and monitor expected-loss drift weekly.",
                "P1: Route borderline cases to assisted review with clear override policy.",
                "P1: Add fairness and policy-compliance checks to monthly model governance cadence.",
            ],
            "insight_method": "Linked validation gains and feature drivers to credit-risk decision controls and operational constraints.",
            "action_method": "Mapped insight-to-policy actions with explicit controls for loss, compliance, and review workload.",
        },
        {
            "industry_name": "Healthcare / Population Risk",
            "keywords": ["health", "obesity", "patient", "clinical", "hospital", "disease"],
            "executive_opening": "This analysis is framed for population-risk management, where intervention timing and capacity planning drive clinical and operational outcomes.",
            "environment_scan": [
                "Industry framing: care delivery setting with high sensitivity to triage quality and intervention timing.",
                "Core KPIs: risk-tier outcome rates, intervention adherence, and avoidable-cost reduction.",
                "Operational rule: prioritize high-risk cohorts while keeping intervention pathways clinically interpretable.",
            ],
            "insight_headlines": [
                "Risk-tiering can improve intervention allocation.",
                "Signal quality supports targeted care pathways.",
                "Deployment scope should match evidence confidence.",
            ],
            "action_hints": [
                "P0: Deploy tiered intervention protocols by predicted risk band and monitor outcome deltas.",
                "P1: Build clinician-facing explanations for top drivers used in triage decisions.",
                "P1: Track subgroup performance drift to maintain equitable intervention quality.",
            ],
            "insight_method": "Integrated validation evidence and driver interpretation to translate prediction quality into intervention planning.",
            "action_method": "Converted risk insights into staged care operations with clear clinical ownership and review cadence.",
        },
        {
            "industry_name": "Manufacturing Quality / Operations",
            "keywords": ["defect", "yield", "line", "plant", "downtime", "inspection", "quality", "machine"],
            "executive_opening": "This analysis is designed for operations teams aiming to reduce quality loss, rework cost, and production disruption.",
            "environment_scan": [
                "Industry framing: throughput-sensitive operations with strict quality and reliability targets.",
                "Core KPIs: defect rate, first-pass yield, downtime, and cost-of-poor-quality.",
                "Operational rule: prioritize interventions where quality-risk reduction materially improves throughput or scrap cost.",
            ],
            "insight_headlines": [
                "Risk scoring can improve inspection prioritization.",
                "Driver-level signals can guide preventive quality controls.",
                "Governance and monitoring determine stable rollout.",
            ],
            "action_hints": [
                "P0: Tighten inspection SOPs on high-risk process windows and monitor first-pass yield changes.",
                "P1: Deploy root-cause checks tied to top drivers and assign owner-by-shift accountability.",
                "P1: Add weekly drift and false-alarm monitoring to avoid unnecessary line interruptions.",
            ],
            "insight_method": "Mapped model signals to process conditions and quality-loss mechanisms.",
            "action_method": "Translated prioritized drivers into SOP, ownership, and escalation controls on the shop floor.",
        },
        {
            "industry_name": "Education Outcomes / Student Success",
            "keywords": ["student", "academic", "course", "curricular", "enroll", "semester", "dropout", "graduate"],
            "executive_opening": "This analysis supports education operations teams improving learner outcomes, retention, and progression quality.",
            "environment_scan": [
                "Industry framing: education operations balancing access, support capacity, and completion outcomes.",
                "Core KPIs: progression rate, dropout risk, completion quality, and support efficiency.",
                "Operational rule: prioritize high-impact support actions for cohorts with elevated outcome risk.",
            ],
            "insight_headlines": [
                "Risk segmentation can improve student support allocation.",
                "Driver signals can guide targeted intervention design.",
                "Rollout confidence depends on transparent outcome tracking.",
            ],
            "action_hints": [
                "P0: Trigger targeted advising and support outreach for high-risk cohorts early in the term.",
                "P1: Align support intensity to predicted risk tiers and monitor progression uplift.",
                "P1: Add cohort-level fairness and drift checks in monthly governance reviews.",
            ],
            "insight_method": "Connected prediction evidence to learner journey stages and support-capacity decisions.",
            "action_method": "Translated insights into intervention ownership, cadence, and measurable outcome checkpoints.",
        },
        {
            "industry_name": "Commercial Forecasting / Pricing",
            "keywords": ["price", "sales", "revenue", "demand", "margin", "listing", "inventory", "forecast"],
            "executive_opening": "This analysis supports commercial teams balancing growth, pricing efficiency, and operational constraints.",
            "environment_scan": [
                "Industry framing: demand-sensitive commercial operations with margin and service trade-offs.",
                "Core KPIs: revenue quality, conversion, forecast error, and margin stability.",
                "Operational rule: bind model-driven actions to explicit financial and service guardrails.",
            ],
            "insight_headlines": [
                "Forecast quality can improve pricing and allocation decisions.",
                "Driver interpretation can reduce overreaction to noisy signals.",
                "Staged rollout lowers financial and operational risk.",
            ],
            "action_hints": [
                "P0: Apply threshold-based pricing or allocation actions only when expected gain exceeds risk tolerance.",
                "P1: Use driver-level monitoring to explain forecast shifts in business reviews.",
                "P1: Track decision impact with holdout or phased rollout design.",
            ],
            "insight_method": "Mapped model signals to commercial decision cycles and uncertainty management.",
            "action_method": "Converted forecast evidence into governed pricing/allocation playbooks with KPI checkpoints.",
        },
        {
            "industry_name": "Premium Cross-Border Passenger Transport & Maritime Tourism",
            "keywords": [
                "passenger",
                "cabin",
                "embarked",
                "fare",
                "ticket",
                "pclass",
                "survived",
                "survival",
                "voyage",
                "port",
                "ship",
                "cruise",
                "lifeboat",
            ],
            "executive_opening": "This analysis is positioned for premium cross-border passenger transport and maritime tourism, focusing on safety assurance, risk segmentation, and differentiated service response under constrained emergency resources.",
            "environment_scan": [
                "Industry framing: high-value passenger mobility with strict safety, compliance, and service-level commitments.",
                "Core KPIs: incident survival outcome, emergency response efficiency, service equity across passenger segments, and trust impact.",
                "Operational rule: when emergency resources are constrained, response sequencing must be optimized by passenger risk profile and physical access constraints.",
            ],
            "insight_headlines": [
                "Passenger-segment risk stratification can materially improve emergency readiness.",
                "Cabin/class and access-related features are strong proxies for response inequality risk.",
                "Safety operations should shift from static SOP to voyage-specific dynamic resource planning.",
            ],
            "action_hints": [
                "P0: Upgrade evacuation guidance density and crew deployment in high-risk cabin/service zones.",
                "P1: Launch family-centric safety programs and segment-level drill protocols to improve perceived and actual safety quality.",
                "P1: Use voyage-level passenger-mix simulation to pre-allocate emergency assets and service response capacity.",
            ],
            "insight_method": "Connected passenger demographics, cabin/access features, and outcome metrics to operational safety and service design trade-offs.",
            "action_method": "Mapped risk insights into differentiated emergency SOP, staffing allocation, and premium safety-service positioning.",
        },
    ]
    best_profile: dict[str, Any] | None = None
    best_score = 0
    for profile in profiles:
        keywords = profile["keywords"]
        score = _industry_keyword_score(
            blob=blob_norm,
            keywords=[str(item).lower() for item in keywords],
            column_signal=column_signal,
        )
        if score > best_score:
            best_score = score
            best_profile = profile
    if best_profile and best_score >= 2:
        return {k: v for k, v in best_profile.items() if k != "keywords"}
    return _semantic_fallback_industry_profile(
        blob=blob_norm,
        column_signal=column_signal,
    )


def _artifact_text_signal(artifact_evidence: dict[str, Any]) -> str:
    summaries = artifact_evidence.get("artifact_summaries")
    if not isinstance(summaries, dict):
        return ""
    snippets: list[str] = []
    for summary in summaries.values():
        if not isinstance(summary, dict):
            continue
        snippet = str(summary.get("summary") or summary.get("preview") or "").strip().lower()
        if snippet:
            snippets.append(_excerpt(snippet, max_chars=200))
    return " ".join(snippets)


def _artifact_column_signal(artifact_evidence: dict[str, Any]) -> list[str]:
    summaries = artifact_evidence.get("artifact_summaries")
    if not isinstance(summaries, dict):
        return []
    tokens: list[str] = []
    for summary in summaries.values():
        if not isinstance(summary, dict):
            continue
        columns = summary.get("columns")
        if not isinstance(columns, list):
            continue
        for column_name in columns:
            raw = str(column_name).strip().lower()
            if raw:
                tokens.append(raw)
                tokens.extend(raw.replace("-", "_").split("_"))
    compact = [token for token in tokens if token]
    seen: set[str] = set()
    deduped: list[str] = []
    for token in compact:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def _industry_keyword_score(*, blob: str, keywords: list[str], column_signal: list[str]) -> int:
    score = 0
    column_blob = " ".join(column_signal)
    normalized_column_blob = column_blob.replace("_", "").replace("-", "").replace(" ", "")
    tokens = set(re.split(r"[^a-z0-9]+", blob))
    tokens.discard("")
    for keyword in keywords:
        term = keyword.strip().lower()
        if not term:
            continue
        if term in blob:
            score += 1
        if term in tokens:
            score += 1
        flattened = term.replace(" ", "").replace("_", "").replace("-", "")
        if flattened and flattened in normalized_column_blob:
            score += 2
    # Boost when multiple industry terms co-occur; helps disambiguate broad words.
    matched_terms = [term for term in keywords if term and term in blob]
    if len(matched_terms) >= 3:
        score += 2
    return score


def _semantic_fallback_industry_profile(*, blob: str, column_signal: list[str]) -> dict[str, Any]:
    text = f"{blob} {' '.join(column_signal)}"
    candidate_domains = [
        (
            "Financial Risk Decisioning & Portfolio Control",
            ("credit", "loan", "default", "fraud", "account", "balance", "payment"),
        ),
        (
            "Passenger Mobility Safety & Service Operations",
            ("passenger", "cabin", "fare", "embarked", "voyage", "port", "travel"),
        ),
        (
            "Quality Assurance & Reliability Operations",
            ("defect", "quality", "inspection", "failure", "machine", "plant", "yield"),
        ),
        (
            "Customer Lifecycle Growth & Retention Operations",
            ("customer", "churn", "retention", "subscription", "campaign", "conversion"),
        ),
        (
            "Population Risk Stratification & Intervention Services",
            ("patient", "health", "disease", "obesity", "clinical", "risk"),
        ),
        (
            "Specialized Entity Risk Management & Response Services",
            ("type", "entity", "bone", "flesh", "soul", "hazard", "containment", "threat"),
        ),
    ]
    best_label = "Adaptive Risk Intelligence & Resource Allocation Services"
    best_score = 0
    for label, tokens in candidate_domains:
        score = sum(1 for token in tokens if token in text)
        if score > best_score:
            best_score = score
            best_label = label
    return {
        "industry_name": best_label,
        "executive_opening": (
            f"This analysis is framed for {best_label.lower()}, where prediction quality determines "
            "resource allocation efficiency, risk control quality, and service outcomes."
        ),
        "scenario_definition": "Optimize classification-driven triage and resource deployment under uncertainty.",
        "environment_scan": [
            "Industry framing: constrained-resource operating system balancing cost, quality, and risk exposure.",
            "Core KPIs: intervention impact, response efficiency, and downside-risk control.",
            "Operational rule: prioritize actions by expected impact and enforce monitoring-driven governance.",
        ],
        "operating_rules": [
            "Escalate response intensity by predicted severity or value tier.",
            "Allocate scarce resources to high-impact segments first.",
        ],
        "decision_economics": [
            "Cost lever: reduce wasted response capacity through better targeting.",
            "Value lever: improve outcome quality through segment-specific treatment.",
            "Risk lever: monitor drift and trigger rollback when control limits are breached.",
        ],
        "insight_headlines": [
            "Signal-driven segmentation can materially improve operating decisions.",
            "Feature-level interpretation clarifies where cost and value are created.",
            "Governance rigor determines whether gains persist in production.",
        ],
        "action_hints": [
            "P0: Define severity or value tiers and bind each tier to a clear response protocol.",
            "P1: Connect decision thresholds to ownership, cadence, KPI guardrails, and rollback rules.",
            "P1: Build ongoing drift and outcome monitoring before scaling coverage.",
        ],
        "insight_method": "Synthesized signal semantics, feature importance, and artifact context into an operating-economics interpretation.",
        "action_method": "Converted the interpretation into tiered operating actions with governance controls and measurable outcomes.",
    }


def _build_business_signal_abstraction(
    *,
    task_family: str,
    feature_importance: dict[str, Any],
    artifact_evidence: dict[str, Any],
    business_context: str,
) -> dict[str, Any]:
    top_features = [driver.name for driver in _top_k_features(feature_importance, k=8)]
    column_tokens = _artifact_column_signal(artifact_evidence)
    signal_space = " ".join([business_context.lower(), " ".join(token.lower() for token in column_tokens)])
    feature_mappings: list[dict[str, str]] = []
    axes: list[str] = []
    for feature in top_features:
        feature_lower = feature.lower()
        meaning = "observable condition signal"
        lever = "prioritize intervention intensity by segment"
        if any(token in feature_lower for token in ("price", "fare", "cost", "income", "amount", "balance")):
            meaning = "unit economics or affordability proxy"
            lever = "optimize pricing, subsidy, or offer intensity by expected margin"
            axes.append("economic")
        elif any(token in feature_lower for token in ("age", "tenure", "duration", "time", "month", "year")):
            meaning = "lifecycle or timing signal"
            lever = "adjust intervention timing and service cadence by lifecycle stage"
            axes.append("lifecycle")
        elif any(token in feature_lower for token in ("location", "region", "zone", "port", "cabin", "route")):
            meaning = "access or spatial-exposure signal"
            lever = "rebalance staffing and assets by location-level risk"
            axes.append("spatial")
        elif any(
            token in feature_lower
            for token in ("quality", "defect", "status", "health", "rotting", "bone", "flesh", "soul", "type")
        ):
            meaning = "condition or severity signal"
            lever = "apply severity-tier SOP and specialized handling protocols"
            axes.append("severity")
        elif any(token in feature_lower for token in ("sex", "gender", "family", "household", "children")):
            meaning = "segment vulnerability or preference signal"
            lever = "differentiate service and safety response by segment profile"
            axes.append("segment")
        feature_mappings.append(
            {
                "feature": feature,
                "business_meaning": meaning,
                "decision_lever": lever,
            }
        )

    if "classification" in task_family:
        operating_rule = "Use risk or class thresholds to prioritize finite response resources."
    else:
        operating_rule = "Use predicted magnitude bands to prioritize interventions by expected value."

    value_paths = [
        "Reduce avoidable response cost through better prioritization.",
        "Increase outcome reliability through segment-specific operating policies.",
        "Lower downside risk with monitoring, thresholds, and rollback controls.",
    ]
    if "economic" in axes:
        value_paths.insert(0, "Improve unit economics through targeted treatment intensity.")
    if "severity" in axes:
        value_paths.insert(0, "Convert severity signals into tiered handling and containment efficiency.")

    return {
        "dominant_signal_axes": sorted(set(axes))[:5],
        "operating_rule_hypotheses": [
            operating_rule,
            "Prefer interventions where expected uplift clearly exceeds operational cost.",
        ],
        "decision_economics_hypotheses": value_paths[:4],
        "feature_to_business_mapping": feature_mappings[:6],
        "signal_space_excerpt": _excerpt(signal_space, max_chars=220),
    }


def _compact_feature_importance_for_llm(
    feature_importance: dict[str, Any], *, k: int
) -> dict[str, float]:
    return {
        driver.name: float(driver.importance)
        for driver in _top_k_features(feature_importance, k=max(1, k))
    }


def _compact_artifact_evidence_for_llm(artifact_evidence: dict[str, Any]) -> dict[str, Any]:
    compact_paths = artifact_evidence.get("artifact_paths")
    if isinstance(compact_paths, list):
        compact_paths = compact_paths[:4]
    else:
        compact_paths = []

    summaries = artifact_evidence.get("artifact_summaries")
    compact_summaries: dict[str, Any] = {}
    if isinstance(summaries, dict):
        for rel_path, summary in list(summaries.items())[:4]:
            if not isinstance(summary, dict):
                continue
            item_type = str(summary.get("type", "")).lower()
            payload: dict[str, Any] = {"type": item_type}
            if item_type == "csv":
                payload["rows"] = summary.get("rows")
                columns = summary.get("columns")
                if isinstance(columns, list):
                    payload["columns"] = [str(col) for col in columns[:10]]
            elif item_type == "json":
                keys = summary.get("keys")
                if isinstance(keys, list):
                    payload["keys"] = [str(item) for item in keys[:10]]
                payload["preview"] = _excerpt(str(summary.get("preview", "")), max_chars=220)
            elif item_type == "markdown":
                payload["summary"] = _excerpt(str(summary.get("summary", "")), max_chars=220)
            compact_summaries[str(rel_path)] = payload

    return {
        "artifact_paths": compact_paths,
        "artifact_summaries": compact_summaries,
    }


def _business_llm_timeout_seconds() -> float:
    """Timeout budget for business_translator LLM calls.

    Restored to the original behavior: default cap is 60s.
    The value can be overridden via `AGENT_BUSINESS_TRANSLATOR_TIMEOUT_SEC`,
    but it will be clamped into [10s, 60s].
    """
    raw = os.getenv("AGENT_BUSINESS_TRANSLATOR_TIMEOUT_SEC", "").strip()
    if not raw:
        return 60.0
    if raw.lower() in {"0", "none", "null", "disable", "disabled"}:
        return 60.0
    try:
        parsed = float(raw)
    except ValueError:
        return 60.0
    return min(600.0, max(10.0, parsed))


def _industry_identifier_timeout_seconds() -> float:
    raw = os.getenv("AGENT_BUSINESS_TRANSLATOR_INDUSTRY_TIMEOUT_SEC", "").strip()
    if not raw:
        return 6.0
    try:
        parsed = float(raw)
    except ValueError:
        return 6.0
    return min(10.0, max(2.0, parsed))


def _industry_identifier_min_confidence() -> float:
    raw = os.getenv("AGENT_BUSINESS_TRANSLATOR_INDUSTRY_MIN_CONF", "").strip()
    if not raw:
        return 0.6
    try:
        parsed = float(raw)
    except ValueError:
        return 0.6
    return min(0.95, max(0.0, parsed))


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _complete_json_with_timeout(
    *,
    system_prompt: str,
    user_prompt: str,
    fallback: dict[str, Any],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    estimated_input_tokens = _estimate_tokens_from_messages(input_messages)
    if timeout_seconds is None:
        timeout_seconds = _business_llm_timeout_seconds()
    trace_metadata = {
        "component": "business_translator",
        "timeout_seconds": timeout_seconds,
        "timeout_disabled": timeout_seconds is None,
        "estimated_input_tokens": estimated_input_tokens,
        "prompt_chars": len(system_prompt) + len(user_prompt),
    }
    with with_tracing_context(tags=["business_translator", "llm", "timeout_guard"], metadata=trace_metadata):
        try:
            coro = complete_json(
                agent_name="business_translator",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=fallback,
            )
            # No code-level timeout: rely on fast model + concise prompt to
            # complete naturally within budget. The `timeout_seconds` value is
            # retained only for tracing/metadata, not enforced via wait_for.
            payload = _run_coro_sync(coro)
        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
            # Timeout still consumes provider-side input processing; record an estimated
            # usage event so local token accounting and visualizations remain meaningful.
            record_token_usage(
                module_name="business_translator",
                input_tokens=estimated_input_tokens,
                output_tokens=0,
                model="timeout_fallback",
                input_messages=input_messages,
                output_content="[fallback due to timeout/cancelled]",
            )
            logger.warning(
                (
                    "business_translator LLM request timed out/cancelled after %.1fs; "
                    "using fallback payload (estimated input tokens=%d)"
                ),
                timeout_seconds,
                estimated_input_tokens,
            )
            return dict(fallback)
        except Exception:
            # Keep token timeline continuous even on non-timeout LLM failures.
            record_token_usage(
                module_name="business_translator",
                input_tokens=estimated_input_tokens,
                output_tokens=0,
                model="error_fallback",
                input_messages=input_messages,
                output_content="[fallback due to llm error]",
            )
            logger.debug("business_translator LLM request failed; using fallback payload", exc_info=True)
            return dict(fallback)
    if isinstance(payload, dict):
        return payload
    return dict(fallback)


def _estimate_tokens_from_messages(messages: list[dict[str, str]]) -> int:
    text = " ".join(str(item.get("content") or "") for item in messages)
    # Heuristic only: approximately 1 token ~= 4 characters for mixed text.
    estimated = max(1, len(text) // 4)
    return estimated


def _extract_artifact_paths_from_technical_report(technical_report_text: str) -> list[str]:
    lines = technical_report_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Artifacts":
            start = idx + 1
            break
    if start is None:
        return []

    paths: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        matches = re.findall(r"`([^`]+)`", stripped)
        if len(matches) >= 2:
            artifact_path = matches[-1].strip()
            if artifact_path and "/" in artifact_path:
                paths.append(artifact_path)

    seen: set[str] = set()
    unique_paths: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def _collect_artifact_evidence(*, workspace_root: str, technical_report_text: str) -> dict[str, Any]:
    artifact_paths = _extract_artifact_paths_from_technical_report(technical_report_text)
    artifact_paths = artifact_paths[:6]
    summaries: dict[str, Any] = {}
    root = Path(workspace_root).resolve()

    for rel_path in artifact_paths:
        full_path = Path(rel_path)
        if not full_path.is_absolute():
            full_path = root / full_path
        if not full_path.exists():
            continue

        suffix = full_path.suffix.lower()
        if suffix == ".md":
            content = _safe_read_text(str(full_path))
            summaries[rel_path] = {
                "type": "markdown",
                "summary": _excerpt(content, max_chars=700),
            }
        elif suffix == ".csv":
            summaries[rel_path] = {"type": "csv", **_summarize_csv(full_path)}
        elif suffix == ".json":
            summaries[rel_path] = {"type": "json", **_summarize_json(full_path)}

    return {
        "artifact_paths": artifact_paths,
        "artifact_summaries": summaries,
    }


def _summarize_csv(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.reader(file_obj)
            header = next(reader, [])
            total_rows = 0
            sample_rows: list[list[str]] = []
            for row in reader:
                total_rows += 1
                if len(sample_rows) < 3:
                    sample_rows.append(row)
                if total_rows >= 5000:
                    break
    except Exception:
        return {"error": "csv_read_failed"}
    if not header:
        return {"rows": 0, "columns": []}
    return {
        "rows": total_rows,
        "columns": header[:20],
        "sample_rows": sample_rows,
    }


def _summarize_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "json_read_failed"}

    if isinstance(data, dict):
        keys = list(data.keys())[:20]
        return {
            "keys": keys,
            "preview": _excerpt(json.dumps(data, ensure_ascii=False), max_chars=500),
        }
    if isinstance(data, list):
        return {
            "length": len(data),
            "preview": _excerpt(json.dumps(data[:3], ensure_ascii=False), max_chars=500),
        }
    return {"preview": _excerpt(str(data), max_chars=1000)}


def _coerce_string(value: Any, *, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _coerce_str_list(value: Any, *, default: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        if cleaned:
            return cleaned
    return list(default)


def _resolve_task_family(
    *,
    task_type: str,
    validation_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> str:
    normalized = task_type.strip().lower()
    classification_aliases = {
        "classification",
        "binary_classification",
        "binary",
        "multiclass_classification",
        "multi_class",
        "multiclass",
    }
    regression_aliases = {"regression", "regressor"}

    if normalized in classification_aliases:
        if "multi" in normalized:
            return "multiclass_classification"
        if "binary" in normalized:
            return "binary_classification"
        return "classification"
    if normalized in regression_aliases:
        return "regression"

    metric_keys = {str(k).lower() for k in {*validation_metrics.keys(), *baseline_metrics.keys()}}
    regression_markers = {"rmse", "mae", "mse", "r2", "mape"}
    classification_markers = {"accuracy", "f1", "precision", "recall", "auc", "roc_auc", "logloss"}

    if metric_keys & regression_markers:
        return "regression"
    if metric_keys & classification_markers:
        return "classification"
    return normalized or "classification"


def _execution_suggestions(
    *,
    task_family: str,
    business_context: str,
    top_drivers: list[FeatureDriver],
) -> list[str]:
    cadence = "weekly" if task_family in {"classification", "binary_classification", "multiclass_classification"} else "bi-weekly"
    owner = "business ops lead + data science owner"
    primary_driver = top_drivers[0].name if top_drivers else "top feature drivers"

    suggestions = [
        f"Owner: {owner}.",
        f"Cadence: {cadence} review of KPI dashboard and model health indicators.",
        f"KPI hints: monitor outcome KPI alongside drift for {primary_driver}.",
    ]
    if business_context:
        suggestions.append(f"Execution scope: align interventions to context '{business_context}'.")
    return suggestions


def _infer_dataset_name(state: GraphState) -> str:
    train_path = str(state.task_config.dataset_paths.train or "").strip()
    if not train_path:
        return ""
    return Path(train_path).parent.name


def _run_coro_sync(coro: Any) -> Any:
    """Run coroutine from sync context (supports existing event loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_queue: Queue[Any] = Queue(maxsize=1)
    error_queue: Queue[BaseException] = Queue(maxsize=1)

    def _runner() -> None:
        # On Windows, ProactorEventLoop raises "Event loop is closed" when
        # httpx/anyio cleanup tasks linger after loop.close().  Use an explicit
        # loop so we can cancel + drain all pending tasks before closing.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_queue.put(loop.run_until_complete(coro))
        except BaseException as exc:  # noqa: BLE001
            error_queue.put(exc)
        finally:
            try:
                # Cancel all lingering tasks (e.g. httpx connection cleanup)
                # and await them so they can finish gracefully before the loop
                # is closed — prevents "Event loop is closed" RuntimeError.
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            finally:
                loop.close()

    # Preserve contextvars (including LangSmith tracing context) across thread hop,
    # so timeout/cancelled child calls stay attached to the main root trace.
    parent_context = contextvars.copy_context()
    thread = threading.Thread(target=lambda: parent_context.run(_runner), daemon=True)
    thread.start()
    thread.join()
    if not error_queue.empty():
        raise error_queue.get()
    return result_queue.get()


def _generate_business_charts(
    *,
    workspace_root: str,
    validation_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    feature_importance: dict[str, Any],
) -> tuple[str, str]:
    metrics_chart = str(
        file_tools.ensure_artifact_parent(
            Path(workspace_root) / "artifacts/figures/business_metrics_comparison.png"
        )
    )
    drivers_chart = str(
        file_tools.ensure_artifact_parent(
            Path(workspace_root) / "artifacts/figures/business_key_drivers.png"
        )
    )
    _write_metrics_chart(metrics_chart, validation_metrics, baseline_metrics)
    _write_drivers_chart(drivers_chart, feature_importance)
    return metrics_chart, drivers_chart


def _write_metrics_chart(
    path: str,
    validation_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    keys = sorted(
        key
        for key in set(validation_metrics.keys()) | set(baseline_metrics.keys())
        if isinstance(validation_metrics.get(key), (int, float))
        or isinstance(baseline_metrics.get(key), (int, float))
    )
    if not keys:
        return

    val_values = [float(validation_metrics.get(k, 0.0) or 0.0) for k in keys]
    base_values = [float(baseline_metrics.get(k, 0.0) or 0.0) for k in keys]

    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    x = range(len(keys))
    width = 0.38
    ax.bar([i - width / 2 for i in x], val_values, width=width, label="Validation")
    ax.bar([i + width / 2 for i in x], base_values, width=width, label="Baseline")
    ax.set_title("Validation vs Baseline Metrics")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_drivers_chart(path: str, feature_importance: dict[str, Any]) -> None:
    drivers = _top_k_features(feature_importance, k=8)
    if not drivers:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [d.name for d in drivers]
    values = [d.importance for d in drivers]
    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    ax.barh(labels, values)
    ax.set_title("Top Business Drivers")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _chart_markdown(path: str | None) -> str:
    if not path:
        return "- (chart not available)"
    return f"![chart]({path})"


def _write_professional_pdf(path: str, markdown: str) -> None:
    """Export a simple professional PDF from markdown text."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
    except Exception:
        return

    target = file_tools.ensure_artifact_parent(path)
    plain_lines = _markdown_to_plain_lines(markdown)
    wrapped: list[str] = []
    for line in plain_lines:
        if not line.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=95) or [""])

    lines_per_page = 42
    pages = [wrapped[i : i + lines_per_page] for i in range(0, len(wrapped), lines_per_page)] or [["Business Report"]]

    with PdfPages(str(target)) as pdf:
        for page_lines in pages:
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
            ax.axis("off")
            y = 0.98
            for line in page_lines:
                safe_line = _ascii_safe_for_pdf(line)
                ax.text(0.0, y, safe_line, fontsize=10, family="DejaVu Sans", va="top")
                y -= 1.0 / lines_per_page
            pdf.savefig(fig)
            plt.close(fig)


def _markdown_to_plain_lines(markdown: str) -> list[str]:
    plain_lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            plain_lines.append(line[2:].upper())
        elif line.startswith("## "):
            plain_lines.append(line[3:].upper())
        elif line.startswith("### "):
            plain_lines.append(line[4:])
        elif line.startswith("- "):
            plain_lines.append(f"• {line[2:]}")
        elif line.startswith("```"):
            continue
        else:
            plain_lines.append(line)
    return plain_lines


def _ascii_safe_for_pdf(text: str) -> str:
    # Matplotlib's default DejaVu Sans lacks full CJK coverage in many envs.
    safe = (text or "").encode("ascii", errors="ignore").decode("ascii")
    safe = re.sub(r"\s{2,}", " ", safe).strip()
    return safe


def _excerpt(text: str, *, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


# ===========================================================================
# region — V3 Schemas
# ===========================================================================

from typing import Literal as _V3Literal  # local alias avoids clashing with existing imports

from pydantic import BaseModel as _V3BaseModel
from pydantic import Field as _V3Field
from pydantic import ConfigDict as _V3ConfigDict
from pydantic import field_validator as _V3field_validator
from pydantic import model_validator as _V3model_validator


class _V3ChartSeries(_V3BaseModel):
    model_config = _V3ConfigDict(extra="forbid")
    name: str
    values: list[float] = _V3Field(min_length=1, max_length=12)

    @_V3field_validator("values", mode="before")
    @classmethod
    def _coerce_xy_pairs(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        coerced = []
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                # LLM returned [x, y] scatter pairs — keep the y-value
                coerced.append(item[1])
            else:
                coerced.append(item)
        return coerced[:12]  # also enforce max_length early


class _V3ChartData(_V3BaseModel):
    model_config = _V3ConfigDict(extra="forbid")
    labels: list[str] = _V3Field(min_length=1, max_length=12)
    series: list[_V3ChartSeries] = _V3Field(min_length=1, max_length=3)

    @_V3field_validator("series", mode="before")
    @classmethod
    def _truncate_series(cls, v: object) -> object:
        if isinstance(v, list):
            # LLM sometimes emits a flat list of numbers instead of [{"name":..., "values":[...]}].
            # Wrap such cases into a single anonymous series.
            if v and all(isinstance(x, (int, float)) for x in v):
                v = [{"name": "value", "values": list(v)}]
            if len(v) > 3:
                return v[:3]
        return v

    @_V3field_validator("labels", mode="before")
    @classmethod
    def _stringify_labels(cls, v: object) -> object:
        if isinstance(v, list):
            return [str(x) if not isinstance(x, str) else x for x in v]
        return v


class _V3ChartSpec(_V3BaseModel):
    model_config = _V3ConfigDict(extra="forbid")
    chart_id: str
    section: _V3Literal["business-context", "key-insights", "action-recommendations"]
    type: _V3Literal["bar", "horizontal_bar", "pie", "line", "scatter", "heatmap"]
    title: str = _V3Field(min_length=1, max_length=200)
    caption: str = _V3Field(min_length=1, max_length=400)
    insight_ref: str = ""
    x_label: str = ""
    y_label: str = ""
    data: _V3ChartData

    @_V3model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, v: object) -> object:
        if isinstance(v, dict):
            v = dict(v)
            for field in ("insight_ref", "x_label", "y_label"):
                if v.get(field) is None:
                    v[field] = ""
            section = v.get("section")
            if isinstance(section, str):
                # Repair common LLM typos / case / underscore variants.
                normalized = section.strip().lower().replace("_", "-")
                aliases = {
                    "key-inssights": "key-insights",
                    "key-insight": "key-insights",
                    "keyinsights": "key-insights",
                    "business-contexts": "business-context",
                    "businesscontext": "business-context",
                    "action-recommendation": "action-recommendations",
                    "actionrecommendations": "action-recommendations",
                }
                v["section"] = aliases.get(normalized, normalized)
        return v


class _V3InsightBlock(_V3BaseModel):
    model_config = _V3ConfigDict(extra="forbid")
    id: str
    title: str
    evidence_html: str
    interpretation_html: str

    @_V3model_validator(mode="before")
    @classmethod
    def _coerce_field_aliases(cls, v: object) -> object:
        if not isinstance(v, dict):
            return v
        v = dict(v)
        if "title" not in v and "title_html" in v:
            v["title"] = v.pop("title_html")
        elif "title_html" in v:
            v.pop("title_html", None)
        for canonical in ("evidence_html", "interpretation_html"):
            alias = canonical[:-len("_html")]
            if canonical not in v and alias in v:
                v[canonical] = v.pop(alias)
            elif alias in v and canonical in v:
                v.pop(alias, None)
        return v


class _V3ActionBlock(_V3BaseModel):
    model_config = _V3ConfigDict(extra="forbid")
    id: str
    title: str
    target_segment_html: str
    diagnosis_html: str
    levers_html: str
    kpi_html: str

    @_V3model_validator(mode="before")
    @classmethod
    def _coerce_field_aliases(cls, v: object) -> object:
        if not isinstance(v, dict):
            return v
        v = dict(v)
        # title <-> title_html: drop _html suffix (title is plain string)
        if "title" not in v and "title_html" in v:
            v["title"] = v.pop("title_html")
        elif "title_html" in v:
            v.pop("title_html", None)
        # For *_html fields, accept the unsuffixed alias if the canonical key is missing.
        for canonical in ("target_segment_html", "diagnosis_html", "levers_html", "kpi_html"):
            alias = canonical[:-len("_html")]
            if canonical not in v and alias in v:
                v[canonical] = v.pop(alias)
            elif alias in v and canonical in v:
                v.pop(alias, None)
        return v


class _V3NinetyDayAction(_V3BaseModel):
    model_config = _V3ConfigDict(extra="forbid")
    action: str
    owner: str
    metric: str
    expected_benefit: str


class _V3ExecutiveBriefing(_V3BaseModel):
    model_config = _V3ConfigDict(extra="forbid")
    judgments_html: str
    ninety_day_table: list[_V3NinetyDayAction] = _V3Field(min_length=1, max_length=8)
    tone_line: str


class _V3BATResponse(_V3BaseModel):
    model_config = _V3ConfigDict(extra="forbid")
    industry_label: str = _V3Field(min_length=1)
    industry_role_voice: str = _V3Field(min_length=1)
    preface_html: str
    business_context_html: str
    insights: list[_V3InsightBlock] = _V3Field(min_length=3, max_length=6)
    actions: list[_V3ActionBlock] = _V3Field(min_length=3, max_length=6)
    executive_briefing: _V3ExecutiveBriefing
    charts: list[_V3ChartSpec] = _V3Field(min_length=0, max_length=5)


# ===========================================================================
# region — V3 HTML Sanitizer
# ===========================================================================

from html.parser import HTMLParser as _V3HTMLParser

_V3_ALLOWED_TAGS = frozenset({
    "p", "h3", "h4", "table", "thead", "tbody", "tr", "td", "th",
    "ol", "ul", "li", "blockquote", "strong", "em", "br",
})
# Attributes are entirely stripped for safety.


class _V3HtmlSanitizer(_V3HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip_depth = 0  # >0 means we are inside a disallowed tag we must drop entirely
        self._dropped_with_content = frozenset({"script", "style", "iframe", "object", "embed"})

    def handle_starttag(self, tag, attrs):
        if tag in self._dropped_with_content:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if tag in _V3_ALLOWED_TAGS:
            self.out.append(f"<{tag}>")
        # Unknown but harmless tags: drop the tag, keep the text inside.

    def handle_endtag(self, tag):
        if tag in self._dropped_with_content:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if tag in _V3_ALLOWED_TAGS:
            self.out.append(f"</{tag}>")

    def handle_startendtag(self, tag, attrs):
        if tag in _V3_ALLOWED_TAGS and self._skip_depth == 0:
            self.out.append(f"<{tag}/>")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        self.out.append(data)


def _v3_inject_preface_question_list(preface_html: str) -> str:
    """If preface lacks an <ol>, extract three operational-decision questions
    embedded in prose and rewrite that paragraph so the questions render as a
    sibling <ol> (red-numbered list) right before the Report perspective.

    Recognized prose patterns:
      - "We must decide: X, Y, and Z."
      - "This report answers: X, Y, Z."
      - A paragraph containing three sentences each ending with "?".
    Returns the input unchanged if no pattern matches.
    """
    import re as _re_local
    if "<ol" in preface_html or "<li" in preface_html:
        return preface_html

    decide_pattern = _re_local.compile(
        r"(<p>)([\s\S]*?)"
        r"((?:we (?:must|need to|should) (?:decide|answer|determine|allocate)|"
        r"this report (?:answers|tackles))[\s:,]+)"
        r"([^<.]+)\.([\s\S]*?)"
        r"(</p>)",
        flags=_re_local.IGNORECASE,
    )

    def _replace_decide(m):
        pre = m.group(2).strip()
        lead = m.group(3).rstrip(",: ").strip()
        body = m.group(4).strip()
        items = _re_local.split(r",\s*(?:and\s+)?|;\s*", body)
        items = [it.strip().rstrip("?").strip() for it in items if len(it.strip()) > 6]
        if len(items) < 3:
            return m.group(0)
        questions = [it[0].upper() + it[1:] + "?" for it in items[:3]]
        ol_html = "<ol>" + "".join(f"<li>{q}</li>" for q in questions) + "</ol>"
        lead_sentence = f"{pre} {lead}:" if pre else f"{lead}:"
        return f"<p>{lead_sentence.strip()}</p>{ol_html}"

    new_html, n = decide_pattern.subn(_replace_decide, preface_html, count=1)
    if n > 0:
        return new_html

    # Fallback: three "?" sentences in a single <p>
    three_q_pattern = _re_local.compile(r"<p>([^<]*?\?[^<]*?\?[^<]*?\?[^<]*?)</p>")

    def _replace_three_q(m):
        text = m.group(1).strip()
        questions = _re_local.findall(r"[A-Z][^?]{6,}?\?", text)
        if len(questions) < 3:
            return m.group(0)
        return "<ol>" + "".join(f"<li>{q.strip()}</li>" for q in questions[:3]) + "</ol>"

    new_html, n = three_q_pattern.subn(_replace_three_q, preface_html, count=1)
    if n > 0:
        return new_html
    return preface_html


def _v3_sanitize_html_fragment(html: str) -> str:
    """Return a sanitized fragment containing only whitelisted tags, no attributes,
    and no <script>/<style>/<iframe>/<object>/<embed> contents."""
    parser = _V3HtmlSanitizer()
    parser.feed(html or "")
    parser.close()
    out = "".join(parser.out)
    import re as _re_local
    # Normalize the "Report perspective:" paragraph: bold ONLY the label, never the body.
    def _normalize_perspective_paragraph(match: "_re_local.Match[str]") -> str:
        body = match.group(1)
        if "Report perspective:" not in body:
            return match.group(0)
        body = _re_local.sub(r"</?strong>", "", body)
        body = _re_local.sub(
            r"\s*Report perspective:\s*",
            "<strong>Report perspective:</strong> ",
            body,
            count=1,
        )
        return f"<p>{body}</p>"

    out = _re_local.sub(
        r"<p>([\s\S]*?)</p>",
        _normalize_perspective_paragraph,
        out,
    )
    return out


# ===========================================================================
# region — V3 Metric-Token Scanner
# ===========================================================================

import re as _v3_re

_V3_METRIC_TOKEN_PATTERNS = [
    (r"\bAUC\b", "AUC"),
    (r"\bF1\b", "F1"),
    (r"\bROC\b", "ROC"),
    (r"\bRMSE\b", "RMSE"),
    (r"\bMAE\b", "MAE"),
    (r"\bMAPE\b", "MAPE"),
    (r"\bR\^?2\b", "R^2"),
    (r"\bprecision\b", "precision"),
    (r"\brecall\b", "recall"),
    (r"\bp-value\b", "p-value"),
    (r"\bfeature importance\b", "feature importance"),
]


def _v3_scan_metric_token_leaks(html_blocks: list[str]) -> list[str]:
    """Return one entry per detected metric-token leak in the format 'token: <snippet>'.

    Business percentages ("64%"), currency amounts ("EUR 1.59B"), and ordinary
    numbers embedded in prose are NOT flagged.
    """
    leaks: list[str] = []
    joined = "\n".join(html_blocks or [])
    for pattern, label in _V3_METRIC_TOKEN_PATTERNS:
        for match in _v3_re.finditer(pattern, joined, flags=_v3_re.IGNORECASE):
            start = max(0, match.start() - 30)
            end = min(len(joined), match.end() + 30)
            snippet = joined[start:end].replace("\n", " ")
            leaks.append(f"{label}: ...{snippet}...")
    return leaks


# ===========================================================================
# region — V3 Fenced JSON Extractor
# ===========================================================================

import json as _v3_json


def _v3_extract_fenced_json(raw: str | None) -> dict | None:
    """Extract a JSON object from a fenced or plain LLM response. Returns None on failure."""
    if not raw:
        return None
    text = raw.strip()
    fenced = _v3_re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=_v3_re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    obj_match = _v3_re.search(r"\{.*\}", text, flags=_v3_re.DOTALL)
    if obj_match:
        text = obj_match.group(0).strip()
    try:
        parsed = _v3_json.loads(text)
    except _v3_json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ===========================================================================
# region — V3 Chart Validator
# ===========================================================================

_V3_DESCRIPTIVE_TITLE_PREFIXES = (
    "distribution of",
    "histogram of",
    "frequency of",
    "breakdown of",
)


def _v3_validate_charts(
    charts: list,                   # list[_V3ChartSpec]
    *,
    insight_ids: set[str],
    action_ids: set[str],
) -> tuple[list, list[str], str | None]:
    """Apply business rules on top of the already-Pydantic-validated chart specs.

    Returns ``(kept, dropped_reasons, abort_reason)``. ``abort_reason`` is non-None
    only when the kept set fails the global floor (>= 2 total AND >= 1 in key-insights).
    """
    kept = []
    dropped: list[str] = []
    valid_refs = insight_ids | action_ids

    for spec in charts:
        # 1. insight_ref must resolve
        if spec.insight_ref not in valid_refs:
            dropped.append(f"{spec.chart_id}: orphan insight_ref={spec.insight_ref}")
            continue
        # 2. title must be a conclusion, not descriptive
        if spec.title.strip().lower().startswith(_V3_DESCRIPTIVE_TITLE_PREFIXES):
            dropped.append(f"{spec.chart_id}: descriptive title rejected")
            continue
        # 3. caption must contain at least one digit
        if not any(ch.isdigit() for ch in spec.caption):
            dropped.append(f"{spec.chart_id}: caption lacks a number")
            continue
        # 4. each series values length must match labels length
        if any(len(s.values) != len(spec.data.labels) for s in spec.data.series):
            dropped.append(f"{spec.chart_id}: series/labels length mismatch")
            continue
        # 5. pie with > 5 slices -> coerce to bar
        if spec.type == "pie" and len(spec.data.labels) > 5:
            coerced = spec.model_copy(update={"type": "bar"})
            kept.append(coerced)
            dropped.append(f"{spec.chart_id}: pie>5 slices coerced to bar")
            continue
        kept.append(spec)

    # 6. Apply per-bucket caps:
    #    - Business Context section: at most 1 chart total
    #    - Each insight or action card: at most 1 chart (keyed by insight_ref)
    capped: list = []
    seen_business_context = False
    seen_refs: set[str] = set()
    for spec in kept:
        if spec.section == "business-context":
            if seen_business_context:
                dropped.append(f"{spec.chart_id}: business-context cap (1) exceeded")
                continue
            seen_business_context = True
            capped.append(spec)
            continue
        # key-insights or action-recommendations: cap per insight_ref
        if spec.insight_ref in seen_refs:
            dropped.append(f"{spec.chart_id}: per-card cap (1) exceeded for ref={spec.insight_ref}")
            continue
        seen_refs.add(spec.insight_ref)
        capped.append(spec)
    kept = capped

    total = len(kept)
    key_insight_count = sum(1 for c in kept if c.section == "key-insights")

    if total < 2 or key_insight_count < 1:
        return (
            kept, dropped,
            f"Insufficient chart evidence: total={total} key_insights={key_insight_count}",
        )
    return kept, dropped, None


# ===========================================================================
# region — V3 Evidence Pack
# ===========================================================================

_V3_PACK_BUDGET_BYTES = 32_768


def _v3_safe_section(label: str, value: object, *, max_chars: int = 2000) -> str:
    """Format a labelled section, hard-truncating its serialized form."""
    try:
        if isinstance(value, str):
            body = value
        else:
            body = _v3_json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        body = repr(value)
    if len(body) > max_chars:
        body = body[:max_chars] + " …[truncated]"
    return f"## {label}\n{body}\n"


def _v3_read_technical_report(state) -> str:
    """Read the technical report markdown — the BAT agent's SOLE data input.

    Per project policy, the business translator must not consume any state field
    beyond the technical report. Runtime fields (workspace_root) are used only to
    locate the output file, not as input data.
    """
    path = (state.report_state.technical_report_path or "").strip()
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


_V3_DATA_PATH_PATTERNS = (
    re.compile(r"(?:^|[\s`'\"/(])data/([A-Za-z0-9_.\-]+)/"),
    re.compile(r"artifacts/runs/([A-Za-z0-9_.\-]+?)_\d{8}_\d{6}_"),
)


def _v3_extract_dataset_name(report_text: str) -> str:
    """Derive the dataset name from references inside the technical report."""
    if not report_text:
        return "dataset"
    for pat in _V3_DATA_PATH_PATTERNS:
        m = pat.search(report_text)
        if m:
            return m.group(1)
    return "dataset"


def _v3_extract_task_family(report_text: str) -> str:
    if not report_text:
        return ""
    m = re.search(r"Task family:\s*`([^`]+)`", report_text)
    return m.group(1).strip().lower() if m else ""


def _v3_extract_limitations(report_text: str) -> list[str]:
    if not report_text:
        return []
    lines = report_text.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == "## Limitations":
            start = i + 1
            break
    if start is None:
        return []
    out: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("- "):
            out.append(stripped[2:].strip())
    return out


_V3_ARTIFACT_PRIORITY = {".md": 0, ".json": 1, ".csv": 2}
_V3_ARTIFACT_MAX_ITEMS = 16
_V3_ARTIFACT_MD_EXCERPT_CHARS = 1800
_V3_ARTIFACT_JSON_PREVIEW_CHARS = 1200
_V3_ARTIFACT_SECTION_CHARS = 2400


def _v3_collect_artifact_evidence(
    *, technical_report_text: str, workspace_root: str
) -> list[tuple[str, dict[str, Any]]]:
    """Read & summarize the attachments referenced by the technical report.

    Paths are taken from the report's ``## Artifacts`` section only — no other
    state field is consulted. Prioritizes narrative MDs, then JSONs, then CSVs.
    """
    paths = _extract_artifact_paths_from_technical_report(technical_report_text)
    paths.sort(key=lambda p: (_V3_ARTIFACT_PRIORITY.get(Path(p).suffix.lower(), 9), p))
    paths = paths[:_V3_ARTIFACT_MAX_ITEMS]
    root = Path(workspace_root).resolve()
    out: list[tuple[str, dict[str, Any]]] = []
    for rel in paths:
        full = Path(rel)
        if not full.is_absolute():
            full = root / full
        if not full.exists():
            continue
        suffix = full.suffix.lower()
        if suffix == ".md":
            content = _safe_read_text(str(full))
            if not content.strip():
                continue
            out.append((rel, {"type": "markdown", "summary": _excerpt(content, max_chars=_V3_ARTIFACT_MD_EXCERPT_CHARS)}))
        elif suffix == ".json":
            summary = _summarize_json(full)
            if "preview" in summary:
                summary["preview"] = _excerpt(str(summary["preview"]), max_chars=_V3_ARTIFACT_JSON_PREVIEW_CHARS)
            out.append((rel, {"type": "json", **summary}))
        elif suffix == ".csv":
            out.append((rel, {"type": "csv", **_summarize_csv(full)}))
    return out


def _v3_build_evidence_pack(
    technical_report_text: str,
    artifact_summaries: list[tuple[str, dict[str, Any]]] | None = None,
) -> str:
    """Evidence pack derived ONLY from the technical report and its attachments.

    Composed of the full report markdown followed by per-attachment summaries
    (markdown excerpts, JSON previews, CSV header+sample rows). Truncated to the
    byte budget at a valid UTF-8 boundary.
    """
    report = (technical_report_text or "").strip()
    if not report:
        return "[Technical report is empty]"
    parts: list[str] = ["## Technical Report\n" + report]
    for rel, summary in artifact_summaries or []:
        parts.append(_v3_safe_section(f"Attachment — {rel}", summary, max_chars=_V3_ARTIFACT_SECTION_CHARS))
    pack = "\n\n".join(parts)
    encoded = pack.encode("utf-8")
    if len(encoded) > _V3_PACK_BUDGET_BYTES:
        encoded = encoded[: _V3_PACK_BUDGET_BYTES - 32]
        pack = encoded.decode("utf-8", errors="ignore") + "\n…[pack truncated]"
    return pack


# ===========================================================================
# region — V3 Prompts (delegated to app.prompts.business_prompt)
# ===========================================================================
#
# The v3 system prompt and user-prompt builder live in
# ``app.prompts.business_prompt``. They are imported at the top of this
# module as ``BUSINESS_TRANSLATOR_SYSTEM_PROMPT`` and
# ``build_business_translation_prompt``.
#
# The private aliases below are kept so existing tests and any downstream
# imports that referenced the v3-internal names continue to work.
_V3_SYSTEM_PROMPT = BUSINESS_TRANSLATOR_SYSTEM_PROMPT
_v3_build_user_prompt = build_business_translation_prompt

# ===========================================================================
# region — V3 LLM Bridge (reuses existing _run_coro_sync helper in this file)
# ===========================================================================


def _v3_call_llm_sync(*, system_prompt: str, user_prompt: str, timeout_seconds: float = 300.0) -> dict:
    """Call the LLM synchronously via the existing _run_coro_sync bridge.

    Returns the parsed dict from complete_json. Raises _V3BusinessTranslatorError
    on timeout, cancellation, or empty/non-dict response.

    Uses a fresh LLMService instance (not the module singleton) so this works
    correctly when called inside an already-running event loop context via
    _run_coro_sync, which spawns a new event loop in a thread.
    """
    from app.llm.agent_config import get_agent_config as _get_agent_config
    from app.llm.schemas import ChatMessage as _ChatMessage
    from app.llm.service import LLMService as _LLMService
    from app.llm.helpers import parse_json_response as _parse_json_response

    cfg = _get_agent_config("business_translator")

    async def _call() -> dict:
        service = _LLMService()
        messages = [
            _ChatMessage(role="system", content=system_prompt),
            _ChatMessage(role="user", content=user_prompt),
        ]
        resp = await service.chat(
            model=cfg.model,
            messages=messages,
            module_name="business_translator",
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stream=False,
        )
        return _parse_json_response(resp.content, fallback={})

    try:
        # No code-level timeout enforced (per project requirement). The
        # `timeout_seconds` parameter is kept for signature compatibility only.
        result = _run_coro_sync(_call())
    except Exception as exc:
        raise _V3BusinessTranslatorError(f"LLM call failed: {exc}") from exc

    if not isinstance(result, dict) or not result:
        raise _V3BusinessTranslatorError("LLM returned empty or non-dict response")
    return result


# ===========================================================================
# region — V3 HTML Renderer
# ===========================================================================

from functools import lru_cache as _v3_lru_cache


@_v3_lru_cache(maxsize=1)
def _v3_read_vendored_chartjs() -> str:
    path = Path(__file__).parent / "vendor" / "chart.umd.min.js"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Vendored Chart.js not found at %s; business report will render without charts.", path)
        return ""


def _v3_ninety_day_table_html(rows) -> str:
    cells = "\n".join(
        f"<tr><td>{i+1}</td><td>{r.action}</td><td>{r.owner}</td><td>{r.metric}</td><td>{r.expected_benefit}</td></tr>"
        for i, r in enumerate(rows)
    )
    return (
        '<table class="ninety-day-table">'
        '<thead><tr><th>#</th><th>Action</th><th>Owner</th><th>Metric</th><th>Expected first-year benefit</th></tr></thead>'
        f'<tbody>{cells}</tbody></table>'
    )


def _v3_chart_figure_html(c) -> str:
    payload = _v3_json.dumps({
        "type": c.type,
        "title": c.title,
        "x_label": c.x_label,
        "y_label": c.y_label,
        "data": c.data.model_dump(),
    }, ensure_ascii=False)
    return (
        f'<figure class="chart" id="chart-{c.chart_id}" data-insight-ref="{c.insight_ref}">'
        f'<div class="chart-canvas"><canvas id="canvas-{c.chart_id}"></canvas></div>'
        f'<figcaption>{c.title}. {c.caption}</figcaption>'
        f'</figure>'
        f'<script type="application/json" id="data-{c.chart_id}">{payload}</script>'
    )


def _v3_insight_card_html(i) -> str:
    return (
        f'<article class="insight-card" id="{i.id}">'
        f'<h3>{i.title}</h3>'
        f'{i.evidence_html}'
        f'{i.interpretation_html}'
        f'</article>'
    )


def _v3_action_card_html(a) -> str:
    return (
        f'<article class="action-card" id="{a.id}">'
        f'<h3>{a.title}</h3>'
        f'<h4>Target segment</h4>{a.target_segment_html}'
        f'<h4>Diagnosis</h4>{a.diagnosis_html}'
        f'<h4>Action levers</h4>{a.levers_html}'
        f'<h4>KPI</h4>{a.kpi_html}'
        f'</article>'
    )


# Regular (non-f) triple-quoted strings — literal `{` and `}` need no escaping here.
_V3_HTML_STYLES = """\
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300..800;1,6..72,300..800&family=Source+Serif+4:ital,opsz,wght@0,8..60,300..700;1,8..60,300..700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
  --paper: #fff1e5;
  --paper-surface: #fff8ef;
  --paper-deep: #f7e3cc;
  --ink: #1a1413;
  --ink-soft: #3a2f28;
  --ink-muted: #6b5d52;
  --accent: #8b1a1a;
  --accent-deep: #5e0e0e;
  --rule: #e7d5c1;
  --rule-strong: #d2b896;
  --gain: #1f7a4d;
  --loss: #b32d2d;
  --slate: #2d3e50;
  --stripe: #f7e9d7;
  --shadow: 0 1px 0 rgba(0,0,0,0.02), 0 14px 32px -22px rgba(60,30,15,0.18);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: 'Source Serif 4', 'Newsreader', Georgia, serif;
  font-optical-sizing: auto;
  color: var(--ink);
  background: var(--paper);
  line-height: 1.62;
  font-size: 16.5px;
  font-feature-settings: "onum" 1, "kern" 1;
  -webkit-font-smoothing: antialiased;
}
::selection { background: rgba(139, 26, 26, 0.18); }

/* ===== HERO / MASTHEAD ===== */
.hero {
  max-width: 1120px;
  margin: 0 auto;
  padding: 52px 56px 32px;
  border-bottom: 2px solid var(--ink);
  position: relative;
}
.hero { text-align: center; }
.hero .industry {
  display: block;
  font-family: 'Newsreader', Georgia, serif;
  font-variation-settings: "opsz" 32;
  font-style: italic;
  font-weight: 400;
  font-size: clamp(16px, 2.93vw, 33.33px);
  letter-spacing: 0;
  text-transform: none;
  color: var(--accent);
  margin: 0 auto 10px;
  max-width: none;
  background: transparent;
  border: 0;
  padding: 0;
  border-radius: 0;
}
.hero .meta { margin-left: auto; margin-right: auto; }
.hero h1 {
  font-family: 'Newsreader', Georgia, serif;
  font-variation-settings: "opsz" 72;
  font-weight: 480;
  font-size: clamp(24px, 4.4vw, 50px);
  line-height: 1.08;
  letter-spacing: -0.014em;
  margin: 10px 0 14px;
  color: var(--ink);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.hero h1 em {
  font-style: normal;
  font-weight: inherit;
  color: inherit;
}
.hero .meta {
  font-family: 'Source Serif 4', Georgia, serif;
  font-style: italic;
  font-size: 15.5px;
  color: var(--ink-soft);
  max-width: 64ch;
  border-top: 1px solid var(--rule);
  padding-top: 14px;
  margin-top: 14px;
}

/* ===== CONTAINER ===== */
.container {
  max-width: 880px;
  margin: 0 auto;
  padding: 20px 56px 96px;
}

/* ===== SECTION HEADS ===== */
section { padding-top: 56px; counter-reset: card; }
.section-caption { display: none; }
h1, h2, h3, h4 { color: var(--ink); }
h2 {
  font-family: 'Newsreader', Georgia, serif;
  font-variation-settings: "opsz" 60;
  font-weight: 480;
  font-size: 36px;
  line-height: 1.1;
  letter-spacing: -0.012em;
  margin: 0 0 20px;
  border: 0;
  padding: 0;
}
h3 {
  font-family: 'Newsreader', Georgia, serif;
  font-variation-settings: "opsz" 28;
  font-weight: 520;
  font-size: 21px;
  line-height: 1.3;
  letter-spacing: -0.005em;
  margin: 22px 0 8px;
}
h4 {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  color: var(--accent);
  margin: 20px 0 6px;
  font-weight: 600;
}

/* ===== BODY TEXT ===== */
p { margin: 10px 0; }
strong { font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums lining-nums; }

/* ===== INSIGHT + ACTION CARDS ===== */
.insight-card, .action-card {
  position: relative;
  background: var(--paper-surface);
  border: 1px solid var(--rule);
  border-top: 3px solid var(--accent);
  padding: 30px 32px 22px;
  margin: 24px 0;
  counter-increment: card;
  box-shadow: var(--shadow);
}
.insight-card::before, .action-card::before {
  content: "INSIGHT  " counter(card, decimal-leading-zero);
  position: absolute;
  top: -3px;
  left: -1px;
  background: var(--accent);
  color: var(--paper);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.22em;
  padding: 4px 11px 3px;
  font-weight: 600;
}
.action-card { border-top-color: var(--ink); }
.action-card::before { content: "ACTION  " counter(card, decimal-leading-zero); background: var(--ink); }
.insight-card h3, .action-card h3 {
  margin: 12px 0 8px;
  font-family: 'Newsreader', Georgia, serif;
  font-variation-settings: "opsz" 36;
  font-style: italic;
  font-weight: 460;
  font-size: 22px;
  line-height: 1.28;
}

/* ===== BLOCKQUOTE + LISTS ===== */
blockquote {
  border: 0;
  border-left: 3px solid var(--accent);
  padding: 4px 0 4px 18px;
  margin: 16px 0;
  font-style: italic;
  color: var(--ink-soft);
  font-size: 17px;
}
ol, ul { padding-left: 22px; margin: 10px 0; }
ol li, ul li { margin: 4px 0; }
ol li::marker { color: var(--accent); font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.92em; }
ul li::marker { color: var(--accent); }

/* ===== TABLES — TABULAR NEWSPAPER STYLE ===== */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0;
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 14.5px;
  font-variant-numeric: tabular-nums lining-nums;
  border-top: 2px solid var(--ink);
  border-bottom: 2px solid var(--ink);
  background: var(--paper-surface);
}
th, td { padding: 10px 14px; text-align: left; vertical-align: top; }
th {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.18em;
  border-bottom: 1px solid var(--ink);
  background: transparent;
}
tbody tr { border-bottom: 1px solid var(--rule); }
tbody tr:nth-child(even) { background: var(--stripe); }
tbody tr:last-child { border-bottom: 0; }

table.ninety-day-table th:first-child,
table.ninety-day-table td:first-child {
  width: 36px;
  color: var(--accent);
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

/* ===== CHARTS AS 'EXHIBITS' ===== */
figure.chart {
  margin: 32px 0;
  padding: 28px 26px 18px;
  background: var(--paper-surface);
  border: 1px solid var(--rule);
  border-top: 2px solid var(--ink);
  position: relative;
  box-shadow: var(--shadow);
}
figure.chart .chart-canvas { position: relative; height: 320px; margin-top: 6px; }
figure.chart figcaption {
  margin-top: 14px;
  padding-top: 10px;
  border-top: 1px solid var(--rule);
  font-family: 'Source Serif 4', Georgia, serif;
  font-style: italic;
  font-size: 13.5px;
  color: var(--ink-soft);
  cursor: pointer;
  transition: color 180ms ease;
}
figure.chart figcaption:hover { color: var(--accent); }

/* ===== TONE CALLOUT — 'CLOSING NOTE' ===== */
.tone-callout {
  position: relative;
  margin: 56px 0 0;
  padding: 44px 48px 38px;
  background: var(--ink);
  color: var(--paper);
  border-top: 4px solid var(--accent);
  font-family: 'Newsreader', Georgia, serif;
  font-variation-settings: "opsz" 60;
  font-style: italic;
  font-weight: 380;
  font-size: 25px;
  line-height: 1.32;
  letter-spacing: -0.005em;
  box-shadow: var(--shadow);
}

/* ===== FOOTER ===== */
footer {
  max-width: 1120px;
  margin: 60px auto 0;
  padding: 20px 56px 56px;
  border-top: 1px solid var(--ink);
  color: var(--ink-muted);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  text-align: center;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 720px) {
  .hero { padding: 36px 22px 24px; }
  .container { padding: 16px 22px 64px; }
  .insight-card, .action-card { padding: 24px 18px 18px; }
  .tone-callout { padding: 28px 22px; font-size: 20px; }
  figure.chart { padding: 22px 16px 14px; }
  footer { padding: 18px 22px 36px; }
}

@media print {
  body { background: #fff; }
  .insight-card, .action-card, figure.chart, .tone-callout { box-shadow: none; }
}
"""


_V3_CHART_BOOTSTRAP_JS = """\
(function() {
  if (typeof Chart === 'undefined') {
    document.querySelectorAll('figure.chart .chart-canvas').forEach(function(node) {
      node.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;border:1px dashed #d2b896;color:#6b5d52;font-family:\\'IBM Plex Mono\\', monospace;font-size:12px;letter-spacing:0.04em;">Chart rendering unavailable in this build</div>';
    });
    return;
  }
  // Bloomberg/FT editorial palette — oxblood lead, ink, slate, gain green,
  // sand-gold, plum, steel, ochre. Used per-series and per-datapoint to give
  // single-series exhibits a chromatic story (one color per bar/slice).
  var palette = ['#8b1a1a', '#1a1413', '#2d3e50', '#1f7a4d', '#c8a951', '#a8556e', '#5a7a8a', '#b67e3d', '#3f5c4a', '#6b4a6f'];
  var rgba = function(hex, a) {
    var h = hex.replace('#', '');
    var r = parseInt(h.substring(0, 2), 16);
    var g = parseInt(h.substring(2, 4), 16);
    var b = parseInt(h.substring(4, 6), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
  };
  Chart.defaults.font.family = "'Source Serif 4', 'IBM Plex Mono', Georgia, serif";
  Chart.defaults.color = '#1b1611';
  document.querySelectorAll('script[type="application/json"][id^="data-"]').forEach(function(node) {
    var spec; try { spec = JSON.parse(node.textContent); } catch (e) { return; }
    var id = node.id.replace(/^data-/, '');
    var canvas = document.getElementById('canvas-' + id);
    if (!canvas) return;
    var chartType = (spec.type === 'horizontal_bar') ? 'bar' : spec.type;
    var multiSeries = spec.data.series.length > 1;
    var perPointTypes = ['bar', 'pie', 'doughnut', 'polarArea'];
    var usePerPoint = !multiSeries && perPointTypes.indexOf(chartType) !== -1;
    var datasets = spec.data.series.map(function(s, idx) {
      if (usePerPoint) {
        var pointColors = s.values.map(function(_, i) { return palette[i % palette.length]; });
        var pointBg = pointColors.map(function(c) { return chartType === 'bar' ? rgba(c, 0.85) : c; });
        return {
          label: s.name,
          data: s.values,
          backgroundColor: pointBg,
          borderColor: pointColors,
          borderWidth: chartType === 'pie' || chartType === 'doughnut' ? 2 : 1,
          hoverOffset: 8,
          fill: true,
        };
      }
      var base = palette[idx % palette.length];
      return {
        label: s.name,
        data: s.values,
        backgroundColor: chartType === 'line' ? rgba(base, 0.18) : rgba(base, 0.85),
        borderColor: base,
        borderWidth: chartType === 'line' ? 2.4 : 1,
        pointBackgroundColor: base,
        pointBorderColor: '#fff8ef',
        pointRadius: chartType === 'line' ? 4 : 0,
        tension: chartType === 'line' ? 0.32 : 0,
        fill: chartType === 'line' ? true : true,
      };
    });
    var config = {
      type: chartType,
      data: { labels: spec.data.labels, datasets: datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        indexAxis: spec.type === 'horizontal_bar' ? 'y' : 'x',
        plugins: { legend: { display: datasets.length > 1, position: 'top', align: 'end' } },
        scales: (spec.type === 'pie') ? {} : {
          x: { grid: { display: false }, ticks: { color: '#3a2f28' }, title: { display: !!spec.x_label, text: spec.x_label, color: '#8b1a1a', font: { size: 10.5, weight: '600' } } },
          y: { grid: { color: '#e7d5c1' }, ticks: { color: '#3a2f28' }, title: { display: !!spec.y_label, text: spec.y_label, color: '#8b1a1a', font: { size: 10.5, weight: '600' } } }
        }
      }
    };
    new Chart(canvas, config);
  });
  document.querySelectorAll('figure.chart figcaption').forEach(function(cap) {
    cap.addEventListener('click', function() {
      var ref = cap.parentElement.getAttribute('data-insight-ref');
      var tgt = ref ? document.getElementById(ref) : null;
      if (tgt) tgt.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
})();
"""


def _v3_report_title(parsed) -> str:
    """Report title shown in <title>/<h1> and used to name the output file."""
    industry = (parsed.industry_label or "").strip()
    base = "Business Recommendations Report"
    return f"{industry} {base}".strip() if industry else base


def _v3_title_to_filename(title: str) -> str:
    """Sanitize a report title into a safe HTML filename."""
    cleaned = re.sub(r"[\\/:*?\"<>|\r\n\t]+", " ", title).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned or "business_report") + ".html"


def _v3_render_html_document(*, parsed, charts) -> str:
    """Assemble the full self-contained HTML report.

    Avoids embedding _V3_HTML_STYLES / _V3_CHART_BOOTSTRAP_JS inside an f-string
    (their JS/CSS bodies contain literal `{` / `}`). Uses string concatenation
    to splice them in instead.
    """
    chartjs_inline = _v3_read_vendored_chartjs()

    def _charts_for(ref_id: str, section: str) -> str:
        return "\n".join(
            _v3_chart_figure_html(c)
            for c in charts
            if c.section == section and c.insight_ref == ref_id
        )

    insight_cards = "\n".join(
        _v3_insight_card_html(i) + _charts_for(i.id, "key-insights")
        for i in parsed.insights
    )
    action_cards = "\n".join(
        _v3_action_card_html(a) + _charts_for(a.id, "action-recommendations")
        for a in parsed.actions
    )

    business_context_charts = "\n".join(_v3_chart_figure_html(c) for c in charts if c.section == "business-context")
    # Stragglers: insight/action-section charts whose insight_ref does not match any card id
    # are appended at the end of their section so they are not silently dropped.
    _matched_insight_chart_ids = {c.chart_id for i in parsed.insights for c in charts if c.section == "key-insights" and c.insight_ref == i.id}
    _matched_action_chart_ids = {c.chart_id for a in parsed.actions for c in charts if c.section == "action-recommendations" and c.insight_ref == a.id}
    key_insight_charts = "\n".join(
        _v3_chart_figure_html(c)
        for c in charts
        if c.section == "key-insights" and c.chart_id not in _matched_insight_chart_ids
    )
    action_charts = "\n".join(
        _v3_chart_figure_html(c)
        for c in charts
        if c.section == "action-recommendations" and c.chart_id not in _matched_action_chart_ids
    )

    table_html = _v3_ninety_day_table_html(parsed.executive_briefing.ninety_day_table)

    report_title = _v3_report_title(parsed)
    head = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        f'<title>{report_title}</title>\n'
        '<style>' + _V3_HTML_STYLES + '</style>\n'
        '</head>\n'
    )

    body_top = f"""<body>
<header class="hero">
  <div class="industry">{parsed.industry_label}</div>
  <h1>Business Recommendations Report</h1>
</header>
<main class="container">
  <section id="preface">
    <div class="section-caption">01 — Preface</div>
    <h2>Preface</h2>
    {parsed.preface_html}
  </section>
  <section id="business-context">
    <div class="section-caption">02 — Business Context</div>
    <h2>Business Context</h2>
    {parsed.business_context_html}
    {business_context_charts}
  </section>
  <section id="key-insights">
    <div class="section-caption">03 — Insight Discovery</div>
    <h2>Insight Discovery</h2>
    {insight_cards}
    {key_insight_charts}
  </section>
  <section id="action-recommendations">
    <div class="section-caption">04 — Action Recommendations</div>
    <h2>Action Recommendations</h2>
    {action_cards}
    {action_charts}
  </section>
  <section id="executive-briefing">
    <div class="section-caption">05 — Executive Briefing</div>
    <h2>Executive Briefing</h2>
    {parsed.executive_briefing.judgments_html}
    {table_html}
    <div class="tone-callout">{parsed.executive_briefing.tone_line}</div>
  </section>
</main>
<footer>Generated by CS33-2 Multi-Agent Collaboration for Automated Data Science Workflows</footer>
"""

    scripts = (
        '<script>' + chartjs_inline + '</script>\n'
        '<script>' + _V3_CHART_BOOTSTRAP_JS + '</script>\n'
        '</body>\n'
        '</html>\n'
    )

    return head + body_top + scripts


# ===========================================================================
# region — V3 Preflight & Success Delta
# ===========================================================================

_V3_API_KEY_ENV_VARS = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEWAPI_API_KEY", "LLM_NEWAPI_API_KEY")


def preflight_business_translator() -> tuple[bool, str | None]:
    """Return ``(ok, reason_or_none)``. Public so external callers (future main.py
    integration, tests) can fail-fast before invoking the agent.

    Accepts any provider: passes preflight if any environment variable whose name
    contains ``API_KEY`` has a non-empty value. The legacy whitelist
    ``_V3_API_KEY_ENV_VARS`` is kept for documentation but is no longer enforced.
    """
    for name, value in os.environ.items():
        if "API_KEY" in name and (value or "").strip():
            return True, None
    return (
        False,
        "Missing API key for business translator. Set any *_API_KEY env var in your .env file.",
    )


def _v3_build_success_delta(
    *,
    html_path: str,
    legacy_business_report_path: str | None,
    industry_label: str,
    charts_count: int,
    limitations_to_append: list[str],
    action_titles: list[str],
    tone_line: str,
) -> dict:
    return {
        "report_state": {
            "business_report_path": html_path,
            "executive_summary": tone_line,
            "recommended_actions": list(action_titles),
            "limitations": list(limitations_to_append),
        },
        "artifacts": {
            "business_report_html": html_path,
            "business_report": html_path,  # repurpose existing v2.1 artifact key
            **(
                {"business_report_legacy_markdown": legacy_business_report_path}
                if legacy_business_report_path
                else {}
            ),
            "business_translator": {
                "status": "completed",
                "error": None,
                "error_class": None,
                "charts_count": charts_count,
                "industry_label": industry_label,
                "format": "html_v3",
            },
        },
    }


# ===========================================================================
# region — V3 Public Orchestrator
# ===========================================================================


def _v3_run_business_translation_impl(state) -> dict:
    # 1. Preflight (zero LLM calls when key missing).
    ok, reason = preflight_business_translator()
    if not ok:
        raise _V3BusinessTranslatorError(reason or "preflight failed")

    # 2. Build prompt + call LLM. The technical report is the SOLE data input;
    # everything passed to the LLM is derived from this single markdown source.
    technical_report_text = _v3_read_technical_report(state)
    if not technical_report_text.strip():
        raise _V3BusinessTranslatorError(
            "Technical report is missing or empty; business translator requires it as the sole data input."
        )
    workspace_root = (state.runtime_state.workspace_root or "").strip() or "."
    artifact_summaries = _v3_collect_artifact_evidence(
        technical_report_text=technical_report_text,
        workspace_root=workspace_root,
    )
    pack = _v3_build_evidence_pack(technical_report_text, artifact_summaries)
    task_family = _v3_extract_task_family(technical_report_text)
    user_prompt = build_business_translation_prompt(evidence_pack=pack, task_family=task_family)
    raw_dict = _v3_call_llm_sync(
        system_prompt=BUSINESS_TRANSLATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        timeout_seconds=300.0,
    )

    # complete_json already parses JSON; if upstream returned a stringified raw payload, re-extract.
    if "industry_label" not in raw_dict and "raw_text" in raw_dict:
        extracted = _v3_extract_fenced_json(raw_dict["raw_text"])
        if extracted is None:
            raise _V3BusinessTranslatorError("LLM response did not match the required JSON contract")
        raw_dict = extracted

    # 3. Pydantic top-level validation.
    try:
        parsed = _V3BATResponse.model_validate(raw_dict)
    except Exception as exc:
        raise _V3BusinessTranslatorError(f"Output schema mismatch: {exc}") from exc

    # 4. Sanitize HTML fragments in-place.
    parsed.preface_html = _v3_sanitize_html_fragment(parsed.preface_html)
    parsed.preface_html = _v3_inject_preface_question_list(parsed.preface_html)
    parsed.business_context_html = _v3_sanitize_html_fragment(parsed.business_context_html)
    for ins in parsed.insights:
        ins.evidence_html = _v3_sanitize_html_fragment(ins.evidence_html)
        ins.interpretation_html = _v3_sanitize_html_fragment(ins.interpretation_html)
    for act in parsed.actions:
        act.target_segment_html = _v3_sanitize_html_fragment(act.target_segment_html)
        act.diagnosis_html = _v3_sanitize_html_fragment(act.diagnosis_html)
        act.levers_html = _v3_sanitize_html_fragment(act.levers_html)
        act.kpi_html = _v3_sanitize_html_fragment(act.kpi_html)
    parsed.executive_briefing.judgments_html = _v3_sanitize_html_fragment(
        parsed.executive_briefing.judgments_html
    )

    # 5. Metric-token leak scan (non-blocking).
    all_blocks = [
        parsed.preface_html,
        parsed.business_context_html,
        parsed.executive_briefing.judgments_html,
        *(i.evidence_html + "\n" + i.interpretation_html for i in parsed.insights),
        *(
            a.target_segment_html + "\n" + a.diagnosis_html + "\n" + a.levers_html + "\n" + a.kpi_html
            for a in parsed.actions
        ),
    ]
    leak_notes = [f"Metric token leaked: {leak}" for leak in _v3_scan_metric_token_leaks(all_blocks)]

    # 6. Chart validation.
    insight_ids = {i.id for i in parsed.insights}
    action_ids = {a.id for a in parsed.actions}
    kept_charts, dropped_reasons, abort_reason = _v3_validate_charts(
        list(parsed.charts), insight_ids=insight_ids, action_ids=action_ids
    )
    if abort_reason is not None:
        raise _V3BusinessTranslatorError(abort_reason)

    # 7. Render HTML and write file.
    html = _v3_render_html_document(parsed=parsed, charts=kept_charts)
    dataset_name = _v3_extract_dataset_name(technical_report_text)
    report_filename = _v3_title_to_filename(f"{dataset_name} business recommendation")
    global_out_dir = Path(workspace_root) / "artifacts" / "reports"
    global_out_dir.mkdir(parents=True, exist_ok=True)
    global_html_path = global_out_dir / report_filename
    global_html_path.write_text(html, encoding="utf-8")

    technical_report_path = Path((state.report_state.technical_report_path or "").strip()).expanduser()
    run_reports_dir: Path | None = None
    if technical_report_path:
        candidate_dir = technical_report_path.parent
        if candidate_dir.name == "reports":
            candidate_dir.mkdir(parents=True, exist_ok=True)
            run_reports_dir = candidate_dir

    html_path = global_html_path
    legacy_business_report_path: Path | None = None
    if run_reports_dir is not None:
        run_html_path = run_reports_dir / "business_report.html"
        run_html_path.write_text(html, encoding="utf-8")
        html_path = run_html_path

        # Keep a lightweight legacy path in the run directory so older callers
        # that still ask for business_report.md resolve inside the same run root.
        legacy_business_report_path = run_reports_dir / "business_report.md"
        legacy_business_report_path.write_text(
            "\n".join(
                [
                    "# Business Report",
                    "",
                    f"Primary HTML report: [{run_html_path.name}]({run_html_path.name})",
                    "",
                    f"Global HTML report: [{global_html_path.name}]({global_html_path})",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    # 8. Build success delta. Limitations also come from the technical report only.
    limitations = list(_v3_extract_limitations(technical_report_text))
    limitations.extend(leak_notes)
    limitations.extend(f"Chart dropped: {r}" for r in dropped_reasons)
    if len(kept_charts) < 3:
        limitations.append(
            f"Only {len(kept_charts)} supporting charts rendered with sufficient evidence."
        )

    return _v3_build_success_delta(
        html_path=str(html_path),
        legacy_business_report_path=str(legacy_business_report_path) if legacy_business_report_path else None,
        industry_label=parsed.industry_label,
        charts_count=len(kept_charts),
        limitations_to_append=limitations,
        action_titles=[a.title for a in parsed.actions],
        tone_line=parsed.executive_briefing.tone_line,
    )
