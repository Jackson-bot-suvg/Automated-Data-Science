"""PlannerAgent execution-plan behavior tests."""

from __future__ import annotations

import json
import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch

from app.agents.planner import PlannerAgent
from app.schemas.state_schema import GraphState
from app.tools.phase_trace import resolve_phase_plan


@pytest.mark.asyncio
async def test_planner_run_writes_execution_plan(tmp_path) -> None:
    planner = PlannerAgent()
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"Age": [20, None, 40], "label": [0, 1, 0]}).to_csv(train_path, index=False)
    state = GraphState(current_phase="data_cleaning")
    state.data_state.current_train_path = str(train_path)
    state.target_spec.target_columns = ["label"]
    mocked_payload = {
        "plan_name": "data_cleaning_plan",
        "steps": [
            {
                "tool": "fill_missing",
                "parameters": {"column": "Age", "strategy": "median"},
                "reasoning": "profile",
            }
        ],
    }

    with patch("app.agents.planner.complete_json", new=AsyncMock(return_value=mocked_payload)):
        update = await planner.run(state)

    assert "execution_plan" in update
    assert "current_phase" not in update
    criteria = update["execution_plan"]["success_criteria"]["data_cleaning"]
    assert criteria["agent_plan"]["plan_name"] == "data_cleaning_plan"
    assert criteria["agent_plan"]["steps"][0]["tool"] == "fill_missing"


@pytest.mark.asyncio
async def test_planner_run_filters_disallowed_tool(tmp_path) -> None:
    planner = PlannerAgent()
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"x": [1, None, 3], "label": [0, 1, 0]}).to_csv(train_path, index=False)
    state = GraphState(current_phase="data_cleaning")
    state.data_state.current_train_path = str(train_path)
    state.target_spec.target_columns = ["label"]
    mocked_payload = {
        "plan_name": "data_cleaning_plan",
        "steps": [
            {"tool": "drop_columns", "parameters": {"columns": ["x"]}, "reasoning": "ok"},
            {"tool": "non_existing_tool", "parameters": {}, "reasoning": "bad"},
        ],
    }

    with patch("app.agents.planner.complete_json", new=AsyncMock(return_value=mocked_payload)):
        update = await planner.run(state)

    steps = update["execution_plan"]["success_criteria"]["data_cleaning"]["agent_plan"]["steps"]
    assert len(steps) == 1
    assert steps[0]["tool"] == "drop_columns"


def test_resolve_phase_plan_preserves_explicit_no_action_needed() -> None:
    state = GraphState(current_phase="data_cleaning")
    state.execution_plan.success_criteria = {
        "data_cleaning": {
            "criteria": [],
            "allowed_tools": ["drop_columns", "fill_missing", "clip_outliers"],
            "agent_plan": {
                "plan_name": "data_cleaning_no_action_needed",
                "steps": [],
            },
        }
    }

    plan, meta = resolve_phase_plan(
        state,
        phase="data_cleaning",
        fallback_plan_name="data_cleaning_deterministic_fallback",
    )

    assert plan.plan_name == "data_cleaning_no_action_needed"
    assert plan.steps == []
    assert meta["plan_source"] == "planner_execution_plan"
    assert meta["fallback_reason"] == "no_action_needed"


@pytest.mark.asyncio
async def test_planner_run_marks_data_cleaning_no_action_needed_when_clean_profile(tmp_path) -> None:
    planner = PlannerAgent()
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"Age": [20, 30, 40], "label": [0, 1, 0]}).to_csv(train_path, index=False)
    state = GraphState(current_phase="data_cleaning")
    state.data_state.current_train_path = str(train_path)
    state.target_spec.target_columns = ["label"]
    mocked_payload = {
        "plan_name": "data_cleaning_tool_plan",
        "steps": [],
    }

    with patch("app.agents.planner.complete_json", new=AsyncMock(return_value=mocked_payload)):
        update = await planner.run(state)

    plan = update["execution_plan"]["success_criteria"]["data_cleaning"]["agent_plan"]
    assert plan["plan_name"] == "data_cleaning_no_action_needed"
    assert plan["steps"] == []


def test_planner_retry_context_extracts_latest_retry_and_previous_plan(tmp_path) -> None:
    planner = PlannerAgent()
    plan_path = tmp_path / "feature_engineering.json"
    plan_path.write_text(
        json.dumps(
            {
                "plan_name": "feature_engineering_tool_plan",
                "steps": [
                    {
                        "tool": "encode_categorical",
                        "parameters": {"column": "target", "method": "label"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    retry_context = planner._retry_context_for_phase(
        phase="feature_engineering",
        profile_summary={
            "review_state": {
                "latest_review_status": "retry",
                "pending_review_phase": "feature_engineering",
                "retry_count_by_phase": {"feature_engineering": 1},
                "review_history": [
                    {
                        "phase": "feature_engineering",
                        "status": "retry",
                        "attempt": 1,
                        "blocking_failures": ["target leakage found in selected_features"],
                        "warnings": ["planner emitted duplicate step"],
                    }
                ],
            },
            "phase_plan_records": {"feature_engineering": str(plan_path)},
        },
    )

    assert retry_context is not None
    assert retry_context["attempt"] == 1
    assert retry_context["blocking_failures"] == [
        "target leakage found in selected_features"
    ]
    assert retry_context["previous_plan_name"] == "feature_engineering_tool_plan"
    assert retry_context["previous_plan_steps"][0]["tool"] == "encode_categorical"
