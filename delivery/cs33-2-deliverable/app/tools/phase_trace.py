"""Helpers for phase-plan and phase-review trace artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.state_schema import GraphState
from app.schemas.tools import ToolCall, ToolCallPlan


def phase_record_path(
    *,
    workspace_root: str,
    artifact_root: str,
    category: str,
    phase: str,
) -> str:
    root = Path(artifact_root)
    if not root.is_absolute():
        root = Path(workspace_root or ".").resolve() / root
    target = root / category / f"{phase}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(target)


def write_phase_record(path: str, payload: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return str(target)


def register_phase_artifact(
    artifacts: dict[str, Any],
    *,
    bucket: str,
    phase: str,
    path: str,
) -> dict[str, Any]:
    updated = dict(artifacts or {})
    nested = dict(updated.get(bucket) or {})
    nested[str(phase)] = str(path)
    updated[bucket] = nested
    return updated


def resolve_phase_plan(
    state: GraphState,
    *,
    phase: str,
    fallback_plan_name: str,
) -> tuple[ToolCallPlan, dict[str, Any]]:
    """Return current phase plan from execution_plan or a deterministic fallback."""

    success_criteria = state.execution_plan.success_criteria or {}
    phase_bundle = success_criteria.get(phase) if isinstance(success_criteria, dict) else {}
    phase_bundle = phase_bundle if isinstance(phase_bundle, dict) else {}
    raw_agent_plan = phase_bundle.get("agent_plan")
    allowed_tools = list(phase_bundle.get("allowed_tools") or [])
    criteria = list(phase_bundle.get("criteria") or [])

    if not isinstance(raw_agent_plan, dict):
        return (
            ToolCallPlan(plan_name=fallback_plan_name, steps=[]),
            {
                "plan_source": "deterministic_fallback",
                "fallback_reason": "execution_plan.success_criteria phase bundle missing or invalid",
                "allowed_tools": allowed_tools,
                "criteria": criteria,
            },
        )

    raw_steps = raw_agent_plan.get("steps")
    if not isinstance(raw_steps, list):
        return (
            ToolCallPlan(plan_name=fallback_plan_name, steps=[]),
            {
                "plan_source": "deterministic_fallback",
                "fallback_reason": "execution_plan.agent_plan steps missing or invalid",
                "allowed_tools": allowed_tools,
                "criteria": criteria,
            },
        )

    raw_plan_name = str(raw_agent_plan.get("plan_name") or fallback_plan_name).strip()
    steps: list[ToolCall] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        tool_name = str(raw_step.get("tool") or "").strip()
        if not tool_name:
            continue
        params = raw_step.get("parameters")
        if not isinstance(params, dict):
            params = {}
        reasoning = str(raw_step.get("reasoning") or "").strip() or None
        steps.append(ToolCall(tool=tool_name, parameters=params, reasoning=reasoning))

    if not steps:
        if raw_plan_name == f"{phase}_no_action_needed" or raw_plan_name.endswith(
            "_no_action_needed"
        ):
            return (
                ToolCallPlan(plan_name=raw_plan_name, steps=[]),
                {
                    "plan_source": "planner_execution_plan",
                    "fallback_reason": "no_action_needed",
                    "allowed_tools": allowed_tools,
                    "criteria": criteria,
                },
            )
        return (
            ToolCallPlan(plan_name=fallback_plan_name, steps=[]),
            {
                "plan_source": "deterministic_fallback",
                "fallback_reason": "execution_plan.agent_plan contained no executable steps",
                "allowed_tools": allowed_tools,
                "criteria": criteria,
            },
        )

    return (
        ToolCallPlan(
            plan_name=raw_plan_name,
            steps=steps,
        ),
        {
            "plan_source": "planner_execution_plan",
            "fallback_reason": "",
            "allowed_tools": allowed_tools,
            "criteria": criteria,
        },
    )
