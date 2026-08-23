"""Unit tests for the main-chain PlannerAgent."""

from __future__ import annotations

import pytest
import pandas as pd

from app.agents import planner as planner_module
from app.agents.planner import run_planner
from app.schemas.state_schema import GraphState, ReviewState, TargetSpec


@pytest.mark.asyncio
async def test_planner_initial_call_returns_execution_plan_only(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("PLANNER_LLM_ENABLED", "false")
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"x": [1, None, 3], "label": [0, 1, 0]}).to_csv(train_path, index=False)
    state = GraphState(current_phase="data_cleaning")
    state.data_state.current_train_path = str(train_path)
    state.target_spec = TargetSpec(
        task_family="binary_classification",
        target_mode="single_target",
        target_columns=["label"],
    )

    result = await run_planner(state)

    assert "execution_plan" in result
    assert set(result.keys()) == {"execution_plan"}


@pytest.mark.asyncio
async def test_planner_builds_current_phase_bundle(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PLANNER_LLM_ENABLED", "false")
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"x": [1, None, 3], "label": [0, 1, 0]}).to_csv(train_path, index=False)
    state = GraphState(
        current_phase="data_cleaning",
        review_state=ReviewState(
            latest_review_status="retry",
        ),
    )
    state.data_state.current_train_path = str(train_path)
    state.target_spec = TargetSpec(
        task_family="binary_classification",
        target_mode="single_target",
        target_columns=["label"],
    )

    result = await run_planner(state)

    bundle = result["execution_plan"]["success_criteria"]
    assert bundle["data_cleaning"]["agent_plan"]["plan_name"]


@pytest.mark.asyncio
async def test_planner_rejects_invalid_llm_phase_tools_and_uses_safe_defaults(
    monkeypatch,
    tmp_path,
) -> None:
    async def fake_complete_json(**kwargs):
        return {
            "plan_name": "bad",
            "steps": [
                {"tool": "not_allowed", "parameters": {}, "reasoning": "bad"}
            ],
        }

    monkeypatch.setattr(planner_module, "complete_json", fake_complete_json)
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"x": [1, None, 3], "cat": ["a", None, "b"], "label": [0, 1, 0]}).to_csv(
        train_path,
        index=False,
    )
    state = GraphState(current_phase="data_cleaning")
    state.data_state.current_train_path = str(train_path)
    state.target_spec = TargetSpec(
        task_family="binary_classification",
        target_mode="single_target",
        target_columns=["label"],
    )

    result = await run_planner(state)
    plan = result["execution_plan"]["success_criteria"]["data_cleaning"]["agent_plan"]
    assert plan["steps"]
    assert all(step["tool"] != "not_allowed" for step in plan["steps"])


@pytest.mark.asyncio
async def test_planner_uses_target_spec_for_allowlist(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PLANNER_LLM_ENABLED", "false")
    train_path = tmp_path / "engineered_train.csv"
    pd.DataFrame({"f1": [1.0, 2.0, 3.0], "label": [0, 1, 0]}).to_csv(train_path, index=False)
    state = GraphState(
        current_phase="model_building",
        review_state=ReviewState(latest_review_status="approved"),
        target_spec=TargetSpec(
            task_family="multiclass_classification",
            target_mode="single_target",
            target_columns=["label"],
        ),
        task_config={"task_type": "regression"},
    )
    state.feature_state.engineered_train_path = str(train_path)

    result = await run_planner(state)

    model_building_plan = result["execution_plan"]["success_criteria"]["model_building"]
    assert "get_candidate_models" in model_building_plan["allowed_tools"]
    assert "compute_metric_bundle" in model_building_plan["allowed_tools"]
    assert any(
        step["tool"] == "configure_scoring"
        for step in model_building_plan["agent_plan"]["steps"]
    )


@pytest.mark.asyncio
async def test_planner_skips_llm_for_data_cleaning_and_uses_deterministic_steps(
    monkeypatch,
    tmp_path,
) -> None:
    async def fake_complete_json(**kwargs):
        raise AssertionError("data_cleaning should not call complete_json")

    monkeypatch.setattr(planner_module, "complete_json", fake_complete_json)
    train_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "mostly_missing": [None, None, 1.0],
            "sometimes_missing": [1.0, None, 3.0],
            "label": [0, 1, 0],
        }
    ).to_csv(train_path, index=False)
    state = GraphState(current_phase="data_cleaning")
    state.data_state.current_train_path = str(train_path)
    state.target_spec = TargetSpec(
        task_family="binary_classification",
        target_mode="single_target",
        target_columns=["label"],
    )

    result = await run_planner(state)

    plan = result["execution_plan"]["success_criteria"]["data_cleaning"]["agent_plan"]
    assert plan["plan_name"] == "data_cleaning_classification_deterministic_default"
    assert plan["steps"]
    assert all(step["tool"] == "fill_missing" for step in plan["steps"])


@pytest.mark.asyncio
async def test_planner_skips_llm_for_indepth_eda_and_uses_deterministic_steps(
    monkeypatch,
    tmp_path,
) -> None:
    async def fake_complete_json(**kwargs):
        raise AssertionError("indepth_eda should not call complete_json")

    monkeypatch.setattr(planner_module, "complete_json", fake_complete_json)
    train_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0],
            "segment": ["a", "b", "a"],
            "label": [0, 1, 0],
        }
    ).to_csv(train_path, index=False)
    state = GraphState(current_phase="indepth_eda")
    state.data_state.current_train_path = str(train_path)
    state.target_spec = TargetSpec(
        task_family="binary_classification",
        target_mode="single_target",
        target_columns=["label"],
    )

    result = await run_planner(state)

    plan = result["execution_plan"]["success_criteria"]["indepth_eda"]["agent_plan"]
    assert plan["plan_name"] == "indepth_eda_classification_deterministic_default"
    assert [step["tool"] for step in plan["steps"]] == [
        "target_analysis",
        "feature_feature_relationships",
        "outlier_analysis",
        "segment_analysis",
    ]
