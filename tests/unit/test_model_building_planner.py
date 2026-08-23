"""Integration tests for LLM-driven model_building planning.

验证 BasePlanner.plan_model_building 返回的 ToolCallPlan 能被 agent 正确
映射到 ModelingConfig；LLM 失败时 agent 走默认 plan 仍可产出 artifacts。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.agents.model_building import (
    ModelingConfig,
    _apply_plan_to_config,
    ResolvedTargetScope,
    run_model_building,
)
from app.schemas.state_schema import (
    EvaluationSpec,
    FeatureState,
    GraphState,
    SubmissionSpec,
    TargetSpec,
)
from app.schemas.tools import ToolCall, ToolCallPlan


# ──────────────────────────────────────────────
# Plan → ModelingConfig merge
# ──────────────────────────────────────────────


def _binary_scope() -> ResolvedTargetScope:
    return ResolvedTargetScope(
        target_columns=["Survived"],
        task_family="binary_classification",
        target_mode="single_target",
        prediction_kind="label",
        primary_metric="roc_auc",
        candidate_metrics=[],
        id_columns=["id"],
        output_columns=["Survived"],
        submission_columns=["id", "Survived"],
    )


def _regression_scope() -> ResolvedTargetScope:
    return ResolvedTargetScope(
        target_columns=["SalePrice"],
        task_family="regression",
        target_mode="single_target",
        prediction_kind="continuous",
        primary_metric="neg_root_mean_squared_error",
        candidate_metrics=[],
        id_columns=["Id"],
        output_columns=["SalePrice"],
        submission_columns=["Id", "SalePrice"],
    )


def test_empty_plan_uses_defaults():
    scope = _binary_scope()
    plan = ToolCallPlan(plan_name="empty", steps=[])
    config = _apply_plan_to_config(plan, scope, plan_applied=False)
    assert isinstance(config, ModelingConfig)
    assert config.preprocessing["numeric_impute"] == "median"
    assert config.preprocessing["scaler"] == "standard"
    assert config.cv["strategy"] == "stratified_kfold"
    assert config.scoring["primary_metric"] == "roc_auc"
    assert config.candidate_models is not None
    assert [m["name"] for m in config.candidate_models] == [
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "LightGBM",
    ]
    assert config.plan_applied is False


def test_regression_scope_defaults_use_kfold():
    config = _apply_plan_to_config(
        ToolCallPlan(plan_name="empty", steps=[]),
        _regression_scope(),
        plan_applied=False,
    )
    assert config.cv["strategy"] == "kfold"
    assert config.scoring["primary_metric"] == "neg_root_mean_squared_error"


def test_configure_preprocessing_overrides_defaults():
    plan = ToolCallPlan(
        plan_name="p1",
        steps=[
            ToolCall(
                tool="configure_preprocessing",
                parameters={"scaler": "minmax", "cat_encode": "ordinal"},
                reasoning="small dataset",
            )
        ],
    )
    config = _apply_plan_to_config(plan, _binary_scope(), plan_applied=True)
    assert config.preprocessing["scaler"] == "minmax"
    assert config.preprocessing["cat_encode"] == "ordinal"
    # Unspecified field keeps default
    assert config.preprocessing["numeric_impute"] == "median"


def test_configure_cv_is_ignored():
    plan = ToolCallPlan(
        plan_name="p2",
        steps=[
            ToolCall(
                tool="configure_cv",
                parameters={"strategy": "kfold", "n_splits": 3},
                reasoning="small data",
            )
        ],
    )
    config = _apply_plan_to_config(plan, _binary_scope(), plan_applied=True)
    assert config.cv["strategy"] == "stratified_kfold"
    assert config.cv["n_splits"] == 3


def test_configure_scoring_overrides():
    plan = ToolCallPlan(
        plan_name="p3",
        steps=[
            ToolCall(
                tool="configure_scoring",
                parameters={
                    "primary_metric": "accuracy",
                    "secondary_metrics": ["f1"],
                },
                reasoning="balanced labels",
            )
        ],
    )
    config = _apply_plan_to_config(plan, _binary_scope(), plan_applied=True)
    assert config.scoring["primary_metric"] == "accuracy"
    assert config.scoring["secondary_metrics"] == []


def test_configure_candidate_models_filters_pool():
    plan = ToolCallPlan(
        plan_name="p4",
        steps=[
            ToolCall(
                tool="configure_candidate_models",
                parameters={
                    "models": [
                        {"name": "LogisticRegression", "param_grid": {"C": [1, 5]}},
                        {"name": "RandomForest"},
                    ]
                },
                reasoning="focus on fast, interpretable models",
            )
        ],
    )
    config = _apply_plan_to_config(plan, _binary_scope(), plan_applied=True)
    assert config.candidate_models is not None
    names = [m["name"] for m in config.candidate_models]
    assert "LogisticRegression" in names
    assert "RandomForest" in names
    assert len(names) <= 5


def test_unknown_tool_in_plan_is_ignored():
    plan = ToolCallPlan(
        plan_name="p5",
        steps=[
            ToolCall(
                tool="do_something_wild",
                parameters={"foo": "bar"},
                reasoning="made up tool",
            )
        ],
    )
    config = _apply_plan_to_config(plan, _binary_scope(), plan_applied=True)
    # Nothing crashes; config falls back to defaults
    assert config.preprocessing["numeric_impute"] == "median"


# ──────────────────────────────────────────────
# End-to-end with mocked LLM planner
# ──────────────────────────────────────────────


def _binary_train(n=60):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id": np.arange(n),
        "Survived": rng.integers(0, 2, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Age": rng.uniform(1, 80, n),
        "Fare": rng.uniform(5, 300, n),
    })


def _binary_test(n=20):
    return _binary_train(n).drop(columns=["Survived"])


def _state_for_binary(tmp_path) -> GraphState:
    train = _binary_train()
    test = _binary_test()
    train.to_csv(tmp_path / "tr.csv", index=False)
    test.to_csv(tmp_path / "te.csv", index=False)
    return GraphState(
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["Survived"],
            target_mode="single_target",
            task_family="binary_classification",
            prediction_kind="label",
            submission=SubmissionSpec(
                id_columns=["id"],
                output_columns=["Survived"],
                submission_columns=["id", "Survived"],
            ),
            evaluation=EvaluationSpec(primary_metric="roc_auc"),
        ),
        feature_state=FeatureState(
            engineered_train_path=str(tmp_path / "tr.csv"),
            engineered_test_path=str(tmp_path / "te.csv"),
        ),
    )


def test_llm_plan_applied_via_mock(tmp_path):
    """Mock BasePlanner.plan_model_building → agent should honor plan."""
    state = _state_for_binary(tmp_path)

    mocked_plan = ToolCallPlan(
        plan_name="mocked_binary_plan",
        steps=[
            ToolCall(
                tool="configure_scoring",
                parameters={"primary_metric": "accuracy",
                            "secondary_metrics": ["f1"]},
                reasoning="mocked",
            ),
            ToolCall(
                tool="configure_candidate_models",
                parameters={"models": [{"name": "LogisticRegression"}]},
                reasoning="fast only",
            ),
            ToolCall(
                tool="configure_cv",
                parameters={"n_splits": 3},
                reasoning="small data",
            ),
        ],
    )

    with patch(
        "app.agents.model_building.resolve_phase_plan",
        return_value=(
            mocked_plan,
            {
                "plan_source": "planner_execution_plan",
                "fallback_reason": "",
                "criteria": [],
                "allowed_tools": [],
            },
        ),
    ):
        result = asyncio.get_event_loop().run_until_complete(run_model_building(state))

    meta = result["model_state"]["validation_metrics"]["_meta"]
    assert meta["plan_name"] == "mocked_binary_plan"
    assert meta["plan_applied"] is True
    assert meta["primary_metric"] == "accuracy"
    # LLM restricted candidates to LogisticRegression only
    assert result["model_state"]["candidate_models"] == ["LogisticRegression"]


def test_llm_planner_failure_falls_back_to_default(tmp_path):
    """When LLM planner raises, agent uses default plan and still completes."""
    state = _state_for_binary(tmp_path)

    with patch(
        "app.agents.model_building.resolve_phase_plan",
        return_value=(
            ToolCallPlan(plan_name="fallback_plan", steps=[]),
            {
                "plan_source": "deterministic_fallback",
                "fallback_reason": "planner unavailable",
                "criteria": [],
                "allowed_tools": [],
            },
        ),
    ):
        result = asyncio.get_event_loop().run_until_complete(run_model_building(state))

    meta = result["model_state"]["validation_metrics"]["_meta"]
    assert meta["plan_applied"] is False
    # Submission still produced
    assert Path(result["artifacts"]["prediction_file"]).exists()
    # Flag recorded
    flags = result["data_state"]["data_quality_flags"]
    assert any("deterministic fallback plan" in f for f in flags)
