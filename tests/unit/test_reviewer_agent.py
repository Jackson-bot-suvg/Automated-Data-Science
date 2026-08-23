"""Unit tests for the reviewer agent MVP chain."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents import reviewer_agent as reviewer_agent_module
from app.agents.reviewer import run_reviewer
from app.agents.reviewer_agent import ReviewerAgent
from app.schemas.artifact_schema import DEFAULT_ARTIFACT_PATHS
from app.schemas.reviewer_schema import ReviewCheckResult
from app.schemas.state_schema import (
    DataState,
    FeatureState,
    GraphState,
    ModelState,
    ReviewState,
    TaskConfig,
)


@pytest.mark.asyncio
async def test_data_cleaning_review_approved(tmp_path, monkeypatch) -> None:
    # Disable actual LLM call by making render_text return fallback
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    cleaned_train = tmp_path / "cleaned_train.csv"
    cleaned_test = tmp_path / "cleaned_test.csv"
    cleaning_report = tmp_path / "cleaning_report.md"
    cleaning_plan = tmp_path / "data_cleaning_plan.json"
    pd.DataFrame({"Survived": [0, 1], "Age": [20, 30]}).to_csv(
        cleaned_train, index=False
    )
    pd.DataFrame({"Age": [21, 40]}).to_csv(cleaned_test, index=False)
    cleaning_report.write_text("# Cleaning report\n", encoding="utf-8")
    cleaning_plan.write_text('{"plan_name":"data_cleaning","steps":[]}\n', encoding="utf-8")

    state = GraphState(
        current_phase="data_cleaning",
        task_config=TaskConfig(target_column="Survived"),
        data_state=DataState(
            current_train_path=str(cleaned_train),
            current_test_path=str(cleaned_test),
            cleaning_summary={"drop_missing": {"columns": ["Age"]}},
            data_quality_flags=["cleaned"],
            target_column="Survived",
        ),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "cleaned_train_data": str(cleaned_train),
            "cleaned_test_data": str(cleaned_test),
            "cleaning_report": str(cleaning_report),
            "phase_plan_records": {"data_cleaning": str(cleaning_plan)},
        },
    )

    result = await ReviewerAgent().run(state)

    assert result["review_state"]["latest_review_status"] == "approved"
    assert result["review_state"]["pending_review_phase"] == "data_cleaning"
    assert len(result["review_state"]["review_history"]) == 1
    assert result["runtime_state"]["last_reviewed_phase"] == "data_cleaning"
    assert result["runtime_state"]["interrupted"] is False


@pytest.mark.asyncio
async def test_data_cleaning_deterministic_fallback_retries(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    cleaned_train = tmp_path / "cleaned_train.csv"
    cleaned_test = tmp_path / "cleaned_test.csv"
    cleaning_report = tmp_path / "cleaning_report.md"
    cleaning_plan = tmp_path / "data_cleaning_plan.json"
    pd.DataFrame({"Survived": [0, 1], "Age": [20, 30]}).to_csv(
        cleaned_train, index=False
    )
    pd.DataFrame({"Age": [21, 40]}).to_csv(cleaned_test, index=False)
    cleaning_report.write_text("# Cleaning report\n", encoding="utf-8")
    cleaning_plan.write_text(
        '{"plan_name":"data_cleaning_deterministic_fallback","plan_source":"deterministic_fallback","fallback_reason":"execution_plan.agent_plan contained no executable steps","steps":[]}\n',
        encoding="utf-8",
    )

    state = GraphState(
        current_phase="data_cleaning",
        task_config=TaskConfig(target_column="Survived"),
        data_state=DataState(
            current_train_path=str(cleaned_train),
            current_test_path=str(cleaned_test),
            cleaning_summary={},
            data_quality_flags=[
                "Data cleaning used deterministic fallback plan: execution_plan.agent_plan contained no executable steps"
            ],
            target_column="Survived",
        ),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "cleaned_train_data": str(cleaned_train),
            "cleaned_test_data": str(cleaned_test),
            "cleaning_report": str(cleaning_report),
            "phase_plan_records": {"data_cleaning": str(cleaning_plan)},
        },
    )

    result = await ReviewerAgent().run(state)

    assert result["review_state"]["latest_review_status"] == "retry"
    assert result["review_state"]["retry_count_by_phase"]["data_cleaning"] == 1
    assert any(
        "did not receive an executable planner plan" in failure
        for failure in result["review_state"]["review_history"][0]["blocking_failures"]
    )


@pytest.mark.asyncio
async def test_feature_engineering_column_mismatch_retries(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    engineered_train = tmp_path / "engineered_train.csv"
    engineered_test = tmp_path / "engineered_test.csv"
    feature_summary = tmp_path / "feature_summary.json"
    feature_plan = tmp_path / "feature_engineering_plan.json"
    pd.DataFrame({"Survived": [0, 1], "Age": [20, 30]}).to_csv(
        engineered_train, index=False
    )
    pd.DataFrame({"Fare": [10.0, 20.0]}).to_csv(engineered_test, index=False)
    feature_summary.write_text('{"selected_features": ["Age"]}\n', encoding="utf-8")
    feature_plan.write_text('{"plan_name":"feature_engineering","steps":[]}\n', encoding="utf-8")

    state = GraphState(
        current_phase="feature_engineering",
        task_config=TaskConfig(target_column="Survived"),
        feature_state=FeatureState(
            engineered_train_path=str(engineered_train),
            engineered_test_path=str(engineered_test),
            selected_features=["Age"],
            feature_transform_summary={"encode": "done"},
        ),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "feature_summary": str(feature_summary),
            "phase_plan_records": {"feature_engineering": str(feature_plan)},
        },
    )

    result = await ReviewerAgent().run(state)

    assert result["review_state"]["latest_review_status"] == "retry"
    assert result["review_state"]["retry_count_by_phase"]["feature_engineering"] == 1
    assert (
        "columns mismatch"
        in result["review_state"]["review_history"][0]["blocking_failures"][0]
    )


@pytest.mark.asyncio
async def test_model_building_third_failure_is_fatal(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    state = GraphState(
        current_phase="model_building",
        task_config=TaskConfig(target_column="Survived"),
        model_state=ModelState(
            prediction_path=str(tmp_path / "missing_submission.csv")
        ),
        review_state=ReviewState(retry_count_by_phase={"model_building": 2}),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "model_results": str(tmp_path / "missing_model_results.json"),
            "best_model": str(tmp_path / "missing_best_model.pkl"),
            "prediction_file": str(tmp_path / "missing_submission.csv"),
            "phase_plan_records": {"model_building": str(tmp_path / "missing_model_plan.json")},
        },
    )

    result = await ReviewerAgent().run(state)

    assert result["review_state"]["latest_review_status"] == "fatal"
    assert result["review_state"]["retry_count_by_phase"]["model_building"] == 3
    assert result["runtime_state"]["interrupted"] is True
    assert result["review_state"]["fatal_errors"]


@pytest.mark.asyncio
async def test_run_reviewer_uses_llm_comment_service(tmp_path, monkeypatch) -> None:
    eda_report = tmp_path / "preliminary_eda.md"
    eda_report.write_text("# EDA\n", encoding="utf-8")

    async def fake_render_text(
        *, agent_name, system_prompt, user_prompt, fallback
    ) -> str:
        assert agent_name == "reviewer"
        assert "preliminary_eda" in user_prompt
        return "llm reviewer comment"

    monkeypatch.setattr(reviewer_agent_module, "render_text", fake_render_text)
    state = GraphState(
        current_phase="preliminary_eda",
        data_state=DataState(
            current_train_path=str(tmp_path / "train.csv"),
            schema_summary={"columns": ["Survived", "Age"]},
            eda_findings={
                "dataset_shape": {"rows": 2, "columns": 2},
                "columns": ["Survived", "Age"],
            },
        ),
        artifacts={**DEFAULT_ARTIFACT_PATHS, "preliminary_eda_report": str(eda_report)},
    )

    result = await run_reviewer(state)

    assert result["review_state"]["latest_review_status"] == "approved"
    assert result["review_state"]["last_review_comment"] == "llm reviewer comment"


@pytest.mark.asyncio
async def test_render_text_returns_fallback_when_no_provider_configured(
    monkeypatch,
) -> None:
    from app.llm.helpers import render_text

    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    comment = await render_text(
        agent_name="reviewer",
        system_prompt="system",
        user_prompt="user",
        fallback="fallback comment",
    )

    assert comment == "fallback comment"
