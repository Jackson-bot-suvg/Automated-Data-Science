from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.planner import PlannerAgent
from app.prompts.planner_prompt import build_phase_tolist_prompt


def test_feature_engineering_prompt_is_phase_scoped_and_protects_targets() -> None:
    prompt = build_phase_tolist_prompt(
        phase="feature_engineering",
        normalized_task_type="classification",
        profile_summary={
            "columns": {
                "Cabin": {"type": "categorical"},
                "Age": {"type": "numeric"},
            }
        },
        target_spec_summary={
            "target_columns": ["Transported"],
            "id_columns": ["PassengerId"],
            "task_family": "binary_classification",
            "primary_metric": "accuracy",
            "candidate_metrics": ["accuracy", "roc_auc"],
        },
        retry_context=None,
        allowed_tools=[
            "extract_regex",
            "create_binning",
            "create_interaction",
            "group_rare_categories",
            "encode_categorical",
        ],
        tool_parameter_schemas={},
    )

    assert "Allowed tools (strict, current phase only):" in prompt
    assert "Never transform protected target or id columns." in prompt
    assert "Bad: encoding the target column as a feature." in prompt
    assert "Return an empty steps list only if there is truly no safe actionable step" in prompt


def test_model_building_prompt_includes_multilabel_guidance() -> None:
    prompt = build_phase_tolist_prompt(
        phase="model_building",
        normalized_task_type="multilabel",
        profile_summary={"columns": {"x": {"type": "numeric"}}},
        target_spec_summary={
            "target_columns": ["A", "B"],
            "id_columns": ["id"],
            "task_family": "multilabel_classification",
            "primary_metric": "roc_auc_macro_ovr",
            "candidate_metrics": ["roc_auc_macro_ovr", "f1_micro"],
        },
        retry_context=None,
        allowed_tools=[
            "configure_preprocessing",
            "configure_candidate_models",
            "configure_scoring",
        ],
        tool_parameter_schemas={},
    )

    assert "Select at most 5 conservative multilabel-compatible candidate models" in prompt
    assert "Avoid binary-only metric assumptions" in prompt or "avoid binary-only metric assumptions" in prompt.lower()


def test_model_building_prompt_includes_clustering_guidance() -> None:
    prompt = build_phase_tolist_prompt(
        phase="model_building",
        normalized_task_type="clustering",
        profile_summary={"columns": {"feat_a": {"type": "numeric"}}},
        target_spec_summary={
            "target_columns": [],
            "id_columns": [],
            "task_family": "clustering",
            "primary_metric": "silhouette_score",
            "candidate_metrics": [],
        },
        retry_context=None,
        allowed_tools=[
            "configure_preprocessing",
            "configure_candidate_models",
            "configure_scoring",
        ],
        tool_parameter_schemas={},
    )

    assert "Select at most 2 unsupervised clustering candidates such as KMeans and DBSCAN." in prompt
    assert "Do not emit classification or regression scoring guidance for clustering tasks." in prompt


def test_prompt_includes_structured_retry_context() -> None:
    prompt = build_phase_tolist_prompt(
        phase="feature_engineering",
        normalized_task_type="classification",
        profile_summary={"columns": {"Age": {"type": "numeric"}}},
        target_spec_summary={
            "target_columns": ["target"],
            "id_columns": ["id"],
            "task_family": "binary_classification",
            "primary_metric": "accuracy",
            "candidate_metrics": ["accuracy"],
        },
        retry_context={
            "attempt": 2,
            "last_status": "retry",
            "blocking_failures": ["selected_features included target column"],
            "warnings": ["planner emitted duplicate transform"],
            "previous_plan_name": "feature_engineering_tool_plan",
            "previous_plan_steps": [{"tool": "encode_categorical", "parameters": {"column": "target"}}],
        },
        allowed_tools=["encode_categorical"],
        tool_parameter_schemas={},
    )

    assert "Retry context:" in prompt
    assert "selected_features included target column" in prompt
    assert "feature_engineering_tool_plan" in prompt
    assert "avoid repeating invalid steps from the previous plan" in prompt


def test_planner_normalize_task_type_recognizes_clustering() -> None:
    planner = PlannerAgent()
    assert planner._normalize_task_type("clustering", "single_target") == "clustering"
    assert planner._normalize_task_type("kmeans_clustering", "single_target") == "clustering"


@pytest.mark.asyncio
async def test_planner_filters_protected_or_schema_incompatible_feature_steps() -> None:
    planner = PlannerAgent()
    mocked_payload = {
        "plan_name": "feature_engineering_test",
        "steps": [
            {
                "tool": "encode_categorical",
                "parameters": {"column": "Transported", "method": "label"},
                "reasoning": "invalid protected column",
            },
            {
                "tool": "extract_regex",
                "parameters": {"column": "Age", "pattern": "(\\d+)", "new_col": "AgeToken"},
                "reasoning": "invalid numeric regex",
            },
            {
                "tool": "create_binning",
                "parameters": {"column": "Age", "n_bins": 4, "new_col": "AgeBin"},
                "reasoning": "valid numeric binning",
            },
        ],
    }

    with patch("app.agents.planner.complete_json", new=AsyncMock(return_value=mocked_payload)):
        result = await planner.plan_feature_engineering(
            {
                "columns": {
                    "Age": {"type": "numeric"},
                    "Cabin": {"type": "categorical"},
                    "Transported": {"type": "categorical"},
                }
            },
            {
                "target_columns": ["Transported"],
                "id_columns": ["PassengerId"],
                "task_family": "binary_classification",
                "target_mode": "single_target",
            },
        )

    assert len(result.steps) == 1
    assert result.steps[0].tool == "create_binning"


@pytest.mark.asyncio
async def test_planner_filters_unsupported_model_candidates_for_regression() -> None:
    planner = PlannerAgent()
    mocked_payload = {
        "plan_name": "model_building_test",
        "steps": [
            {
                "tool": "configure_candidate_models",
                "parameters": {"models": [{"name": "LogisticRegression"}]},
                "reasoning": "invalid for regression",
            },
            {
                "tool": "configure_candidate_models",
                "parameters": {"models": [{"name": "Ridge"}, {"name": "RandomForest"}]},
                "reasoning": "valid regression candidates",
            },
        ],
    }

    with patch("app.agents.planner.complete_json", new=AsyncMock(return_value=mocked_payload)):
        result = await planner.plan_model_building(
            {"columns": {"SalePrice": {"type": "numeric"}}},
            {
                "target_columns": ["SalePrice"],
                "id_columns": ["Id"],
                "task_family": "regression",
                "target_mode": "single_target",
                "primary_metric": "neg_root_mean_squared_error",
                "candidate_metrics": ["neg_root_mean_squared_error", "r2"],
            },
        )

    assert len(result.steps) == 1
    assert result.steps[0].parameters["models"] == [
        {"name": "Ridge"},
        {"name": "RandomForest"},
    ]


def test_planner_candidate_model_safety_allows_new_supported_names() -> None:
    planner = PlannerAgent()
    profile_summary = {"columns": {"feat_a": {"type": "numeric"}}}

    assert planner._is_phase_step_safe(
        phase="model_building",
        tool_name="configure_candidate_models",
        params={"models": [{"name": "SVC"}, {"name": "XGBoost"}]},
        profile_summary=profile_summary,
        target_spec_summary={
            "target_columns": ["y"],
            "id_columns": [],
            "prediction_kind": "label",
        },
        normalized_task_type="classification",
    )
    assert planner._is_phase_step_safe(
        phase="model_building",
        tool_name="configure_candidate_models",
        params={"models": [{"name": "KMeans"}, {"name": "DBSCAN"}]},
        profile_summary=profile_summary,
        target_spec_summary={"target_columns": [], "id_columns": []},
        normalized_task_type="clustering",
    )
    assert planner._is_phase_step_safe(
        phase="model_building",
        tool_name="configure_candidate_models",
        params={"models": [{"name": "Ridge"}, {"name": "XGBoost"}]},
        profile_summary=profile_summary,
        target_spec_summary={"target_columns": ["y1", "y2"], "id_columns": []},
        normalized_task_type="multi_output",
    )


def test_planner_candidate_model_safety_rejects_probability_svc_or_large_pool() -> None:
    planner = PlannerAgent()
    profile_summary = {"columns": {"feat_a": {"type": "numeric"}}}

    assert not planner._is_phase_step_safe(
        phase="model_building",
        tool_name="configure_candidate_models",
        params={"models": [{"name": "SVC"}]},
        profile_summary=profile_summary,
        target_spec_summary={
            "target_columns": ["y"],
            "id_columns": [],
            "prediction_kind": "probability",
        },
        normalized_task_type="classification",
    )
    assert not planner._is_phase_step_safe(
        phase="model_building",
        tool_name="configure_candidate_models",
        params={
            "models": [
                {"name": "LogisticRegression"},
                {"name": "RandomForest"},
                {"name": "HistGradientBoosting"},
                {"name": "XGBoost"},
                {"name": "SVC"},
                {"name": "RandomForest"},
            ]
        },
        profile_summary=profile_summary,
        target_spec_summary={"target_columns": ["y"], "id_columns": []},
        normalized_task_type="classification",
    )
