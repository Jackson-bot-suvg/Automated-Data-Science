"""PlannerAgent toollist service tests."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from app.agents.planner import PlannerAgent


@pytest.mark.asyncio
async def test_plan_data_cleaning_returns_tool_call_plan() -> None:
    planner = PlannerAgent()
    mocked_payload = {
        "plan_name": "data_cleaning_test",
        "steps": [
            {
                "tool": "fill_missing",
                "parameters": {"column": "Age", "strategy": "median"},
                "reasoning": "缺失值填充",
            }
        ],
    }

    with patch("app.agents.planner.complete_json", new=AsyncMock(return_value=mocked_payload)):
        result = await planner.plan_data_cleaning(
            {"dataset_shape": {"rows": 100, "columns": 5}},
            {"target_columns": ["Survived"]},
        )

    assert result.plan_name == "data_cleaning_test"
    assert len(result.steps) == 1
    assert result.steps[0].tool == "fill_missing"


@pytest.mark.asyncio
async def test_plan_feature_engineering_filters_unknown_tool() -> None:
    planner = PlannerAgent()
    mocked_payload = {
        "plan_name": "feature_engineering_test",
        "steps": [
            {
                "tool": "not_allowed_tool",
                "parameters": {},
                "reasoning": "invalid",
            },
            {
                "tool": "encode_categorical",
                "parameters": {"column": "Sex", "method": "label"},
                "reasoning": "类别编码",
            },
        ],
    }

    with patch("app.agents.planner.complete_json", new=AsyncMock(return_value=mocked_payload)):
        result = await planner.plan_feature_engineering(
            {"dataset_shape": {"rows": 200, "columns": 10}},
            {"target_columns": ["Survived"]},
        )

    assert result.plan_name == "feature_engineering_test"
    assert len(result.steps) == 1
    assert result.steps[0].tool == "encode_categorical"


@pytest.mark.asyncio
async def test_plan_model_building_uses_configure_tool_family() -> None:
    planner = PlannerAgent()
    mocked_payload = {
        "plan_name": "model_building_test",
        "steps": [
            {
                "tool": "configure_scoring",
                "parameters": {"primary": "roc_auc", "secondary": ["accuracy", "f1"]},
                "reasoning": "配置评估指标",
            },
            {
                "tool": "configure_candidate_models",
                "parameters": {"models": ["RandomForest", "LogisticRegression"]},
                "reasoning": "模型集合",
            },
            {
                "tool": "fit_and_select_model",
                "parameters": {},
                "reasoning": "should be filtered",
            },
        ],
    }

    with patch("app.agents.planner.complete_json", new=AsyncMock(return_value=mocked_payload)):
        result = await planner.plan_model_building(
            {"dataset_shape": {"rows": 200, "columns": 20}},
            {
                "task_family": "binary_classification",
                "target_mode": "single_target",
                "target_columns": ["label"],
            },
        )

    assert result.plan_name == "model_building_test"
    assert len(result.steps) == 2
    assert result.steps[0].parameters["primary_metric"] == "roc_auc"
    assert result.steps[1].parameters["models"] == [
        {"name": "RandomForest"},
        {"name": "LogisticRegression"},
    ]
