from __future__ import annotations

import asyncio
import json

import pandas as pd

from app.agents import reviewer_agent as reviewer_agent_module
from app.agents.reviewer_agent import ReviewerAgent
from app.schemas.artifact_schema import DEFAULT_ARTIFACT_PATHS
from app.schemas.state_schema import (
    DataState,
    FeatureState,
    GraphState,
    ModelState,
    ReviewState,
    RuntimeState,
    SubmissionSpec,
    TargetSpec,
)
from app.tools.reviewer_tools import (
    review_data_cleaning,
    review_feature_engineering,
    review_model_building,
)


def _target_spec(
    *,
    target_columns: list[str],
    id_columns: list[str],
    output_columns: list[str] | None = None,
    submission_columns: list[str] | None = None,
    task_family: str = "binary_classification",
    target_mode: str = "single_target",
    prediction_kind: str = "label",
) -> TargetSpec:
    resolved_output_columns = list(output_columns or target_columns)
    resolved_submission_columns = list(
        submission_columns or [*id_columns, *resolved_output_columns]
    )
    return TargetSpec(
        status="resolved",
        target_columns=list(target_columns),
        target_mode=target_mode,
        task_family=task_family,
        prediction_kind=prediction_kind,
        submission=SubmissionSpec(
            id_columns=list(id_columns),
            output_columns=resolved_output_columns,
            submission_columns=resolved_submission_columns,
        ),
    )


def test_review_data_cleaning_accepts_multi_target_contract(tmp_path) -> None:
    cleaned_train = tmp_path / "cleaned_train.csv"
    cleaned_test = tmp_path / "cleaned_test.csv"
    cleaning_report = tmp_path / "cleaning_report.md"
    cleaning_plan = tmp_path / "data_cleaning_plan.json"
    pd.DataFrame(
        {
            "ImageId": [1, 2],
            "feature_a": [0.3, 0.5],
            "Defect_1": [0, 1],
            "Defect_2": [1, 0],
        }
    ).to_csv(cleaned_train, index=False)
    pd.DataFrame(
        {
            "ImageId": [3, 4],
            "feature_a": [0.2, 0.4],
        }
    ).to_csv(cleaned_test, index=False)
    cleaning_report.write_text("# cleaning\n", encoding="utf-8")
    cleaning_plan.write_text(
        '{"plan_name":"data_cleaning","plan_source":"planner_execution_plan","steps":[{"tool":"fill_missing","parameters":{"column":"feature_a","strategy":"median"}}]}\n',
        encoding="utf-8",
    )

    state = GraphState(
        target_spec=_target_spec(
            target_columns=["Defect_1", "Defect_2"],
            id_columns=["ImageId"],
            task_family="multilabel_classification",
            target_mode="multi_label",
            prediction_kind="probability",
        ),
        data_state=DataState(
            current_train_path=str(cleaned_train),
            current_test_path=str(cleaned_test),
            cleaning_summary={"fill_missing": {"feature_a": "median"}},
            data_quality_flags=["checked"],
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "cleaned_train_data": str(cleaned_train),
            "cleaned_test_data": str(cleaned_test),
            "cleaning_report": str(cleaning_report),
            "phase_plan_records": {"data_cleaning": str(cleaning_plan)},
        },
    )

    result = review_data_cleaning(state)

    assert result.passed


def test_review_data_cleaning_retries_when_planner_fell_back_without_steps(
    tmp_path,
) -> None:
    cleaned_train = tmp_path / "cleaned_train.csv"
    cleaned_test = tmp_path / "cleaned_test.csv"
    cleaning_report = tmp_path / "cleaning_report.md"
    cleaning_plan = tmp_path / "data_cleaning_plan.json"
    pd.DataFrame(
        {
            "id": [1, 2],
            "feature_a": [0.3, 0.5],
            "target": [0, 1],
        }
    ).to_csv(cleaned_train, index=False)
    pd.DataFrame({"id": [3, 4], "feature_a": [0.2, 0.4]}).to_csv(
        cleaned_test, index=False
    )
    cleaning_report.write_text("# cleaning\n", encoding="utf-8")
    cleaning_plan.write_text(
        json.dumps(
            {
                "plan_name": "data_cleaning_deterministic_fallback",
                "plan_source": "deterministic_fallback",
                "fallback_reason": "execution_plan.agent_plan contained no executable steps",
                "steps": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = GraphState(
        target_spec=_target_spec(
            target_columns=["target"],
            id_columns=["id"],
            task_family="binary_classification",
            target_mode="single_target",
        ),
        data_state=DataState(
            current_train_path=str(cleaned_train),
            current_test_path=str(cleaned_test),
            cleaning_summary={},
            data_quality_flags=[
                "Data cleaning used deterministic fallback plan: execution_plan.agent_plan contained no executable steps"
            ],
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "cleaned_train_data": str(cleaned_train),
            "cleaned_test_data": str(cleaned_test),
            "cleaning_report": str(cleaning_report),
            "phase_plan_records": {"data_cleaning": str(cleaning_plan)},
        },
    )

    result = review_data_cleaning(state)

    assert not result.passed
    assert any(
        "did not receive an executable planner plan" in failure
        for failure in result.blocking_failures
    )


def test_review_feature_engineering_blocks_when_engineered_test_leaks_target(
    tmp_path,
) -> None:
    engineered_train = tmp_path / "engineered_train.csv"
    engineered_test = tmp_path / "engineered_test.csv"
    feature_summary = tmp_path / "feature_summary.json"
    pd.DataFrame(
        {
            "ImageId": [1, 2],
            "feature_a": [0.3, 0.5],
            "feature_b": [1.0, 0.4],
            "Defect_1": [0, 1],
            "Defect_2": [1, 0],
        }
    ).to_csv(engineered_train, index=False)
    pd.DataFrame(
        {
            "ImageId": [3, 4],
            "feature_a": [0.2, 0.4],
            "feature_b": [0.8, 0.6],
            "Defect_2": [0, 1],
        }
    ).to_csv(engineered_test, index=False)
    feature_summary.write_text('{"selected_features": ["feature_a", "feature_b"]}\n', encoding="utf-8")

    state = GraphState(
        target_spec=_target_spec(
            target_columns=["Defect_1", "Defect_2"],
            id_columns=["ImageId"],
            task_family="multilabel_classification",
            target_mode="multi_label",
            prediction_kind="probability",
        ),
        feature_state=FeatureState(
            engineered_train_path=str(engineered_train),
            engineered_test_path=str(engineered_test),
            selected_features=["feature_a", "feature_b"],
            feature_transform_summary={"plan": "ok"},
            leakage_checks=["target columns removed from test"],
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        artifacts={**DEFAULT_ARTIFACT_PATHS, "feature_summary": str(feature_summary)},
    )

    result = review_feature_engineering(state)

    assert not result.passed
    assert any(
        "still contains target columns Defect_2" in failure
        for failure in result.blocking_failures
    )


def test_review_model_building_accepts_submission_contract_across_task_shapes(
    tmp_path,
) -> None:
    cases = [
        {
            "name": "binary",
            "target_spec": _target_spec(
                target_columns=["Exited"],
                id_columns=["id"],
                task_family="binary_classification",
                target_mode="single_target",
                prediction_kind="probability",
            ),
            "prediction": pd.DataFrame({"id": [1, 2], "Exited": [0.2, 0.8]}),
        },
        {
            "name": "regression",
            "target_spec": _target_spec(
                target_columns=["SalePrice"],
                id_columns=["Id"],
                task_family="regression",
                target_mode="single_target",
                prediction_kind="continuous",
            ),
            "prediction": pd.DataFrame({"Id": [1, 2], "SalePrice": [120000.0, 150000.0]}),
        },
        {
            "name": "multiclass_probability",
            "target_spec": _target_spec(
                target_columns=["Cover_Type"],
                id_columns=["Id"],
                output_columns=["Class_1", "Class_2", "Class_3"],
                submission_columns=["Id", "Class_1", "Class_2", "Class_3"],
                task_family="multiclass_classification",
                target_mode="single_target",
                prediction_kind="probability",
            ),
            "prediction": pd.DataFrame(
                {
                    "Id": [1, 2],
                    "Class_1": [0.1, 0.2],
                    "Class_2": [0.7, 0.5],
                    "Class_3": [0.2, 0.3],
                }
            ),
        },
        {
            "name": "multi_target",
            "target_spec": _target_spec(
                target_columns=["Defect_1", "Defect_2"],
                id_columns=["ImageId"],
                task_family="multilabel_classification",
                target_mode="multi_label",
                prediction_kind="probability",
            ),
            "prediction": pd.DataFrame(
                {
                    "ImageId": [1, 2],
                    "Defect_1": [0.1, 0.2],
                    "Defect_2": [0.8, 0.4],
                }
            ),
        },
    ]

    for case in cases:
        case_dir = tmp_path / case["name"]
        case_dir.mkdir()
        engineered_test = case_dir / "engineered_test.csv"
        prediction_file = case_dir / "submission.csv"
        model_results = case_dir / "model_results.json"
        best_model = case_dir / "best_model.pkl"
        model_plan = case_dir / "model_building_plan.json"

        id_columns = case["target_spec"].submission.id_columns
        pd.DataFrame({id_columns[0]: [1, 2], "feature_a": [0.3, 0.5]}).to_csv(
            engineered_test, index=False
        )
        case["prediction"].to_csv(prediction_file, index=False)
        model_results.write_text("{}", encoding="utf-8")
        best_model.write_bytes(b"model")
        model_plan.write_text(
            '{"plan_name":"model_building","plan_source":"planner_execution_plan","steps":[{"tool":"configure_scoring","parameters":{"primary_metric":"accuracy"}}]}\n',
            encoding="utf-8",
        )

        state = GraphState(
            target_spec=case["target_spec"],
            feature_state=FeatureState(engineered_test_path=str(engineered_test)),
            model_state=ModelState(
                candidate_models=["ModelA"],
                best_model_name="ModelA",
                validation_metrics={"score": 1.0},
                cv_results={"ModelA": {"score": 1.0}},
                prediction_path=str(prediction_file),
                baseline_metrics={"score": 1.0},
            ),
            runtime_state=RuntimeState(workspace_root=str(case_dir)),
            artifacts={
                **DEFAULT_ARTIFACT_PATHS,
                "model_results": str(model_results),
                "best_model": str(best_model),
                "prediction_file": str(prediction_file),
                "phase_plan_records": {"model_building": str(model_plan)},
            },
        )

        result = review_model_building(state)
        assert result.passed, (case["name"], result.blocking_failures)


def test_review_model_building_blocks_on_submission_order_and_nulls(tmp_path) -> None:
    engineered_test = tmp_path / "engineered_test.csv"
    prediction_file = tmp_path / "submission.csv"
    model_results = tmp_path / "model_results.json"
    best_model = tmp_path / "best_model.pkl"
    model_plan = tmp_path / "model_building_plan.json"
    pd.DataFrame({"ImageId": [1, 2], "feature_a": [0.3, 0.5]}).to_csv(
        engineered_test, index=False
    )
    pd.DataFrame(
        {
            "Defect_1": [0.1, 0.2],
            "ImageId": [1, 2],
            "Defect_2": [0.8, None],
        }
    ).to_csv(prediction_file, index=False)
    model_results.write_text("{}", encoding="utf-8")
    best_model.write_bytes(b"model")
    model_plan.write_text(
        '{"plan_name":"model_building","plan_source":"planner_execution_plan","steps":[{"tool":"configure_scoring","parameters":{"primary_metric":"roc_auc_macro_ovr"}}]}\n',
        encoding="utf-8",
    )

    state = GraphState(
        target_spec=_target_spec(
            target_columns=["Defect_1", "Defect_2"],
            id_columns=["ImageId"],
            task_family="multilabel_classification",
            target_mode="multi_label",
            prediction_kind="probability",
        ),
        feature_state=FeatureState(engineered_test_path=str(engineered_test)),
        model_state=ModelState(
            candidate_models=["ModelA"],
            best_model_name="ModelA",
            validation_metrics={"score": 1.0},
            cv_results={"ModelA": {"score": 1.0}},
            prediction_path=str(prediction_file),
            baseline_metrics={"score": 1.0},
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "model_results": str(model_results),
            "best_model": str(best_model),
            "prediction_file": str(prediction_file),
            "phase_plan_records": {"model_building": str(model_plan)},
        },
    )

    result = review_model_building(state)

    assert not result.passed
    assert any(
        "columns do not match target_spec submission contract" in failure
        for failure in result.blocking_failures
    )
    assert any("contains null values" in failure for failure in result.blocking_failures)


def test_reviewer_agent_approves_multiclass_probability_submission(
    tmp_path, monkeypatch
) -> None:
    async def fake_render_text(*, agent_name, system_prompt, user_prompt, fallback) -> str:
        return fallback

    monkeypatch.setattr(reviewer_agent_module, "render_text", fake_render_text)

    engineered_test = tmp_path / "engineered_test.csv"
    prediction_file = tmp_path / "submission.csv"
    model_results = tmp_path / "model_results.json"
    best_model = tmp_path / "best_model.pkl"
    model_plan = tmp_path / "model_building_plan.json"
    pd.DataFrame({"Id": [1, 2], "feature_a": [0.3, 0.5]}).to_csv(
        engineered_test, index=False
    )
    pd.DataFrame(
        {
            "Id": [1, 2],
            "Class_1": [0.1, 0.2],
            "Class_2": [0.7, 0.5],
            "Class_3": [0.2, 0.3],
        }
    ).to_csv(prediction_file, index=False)
    model_results.write_text("{}", encoding="utf-8")
    best_model.write_bytes(b"model")
    model_plan.write_text(
        '{"plan_name":"model_building","plan_source":"planner_execution_plan","steps":[{"tool":"configure_scoring","parameters":{"primary_metric":"accuracy"}}]}\n',
        encoding="utf-8",
    )

    state = GraphState(
        current_phase="model_building",
        target_spec=_target_spec(
            target_columns=["Cover_Type"],
            id_columns=["Id"],
            output_columns=["Class_1", "Class_2", "Class_3"],
            submission_columns=["Id", "Class_1", "Class_2", "Class_3"],
            task_family="multiclass_classification",
            target_mode="single_target",
            prediction_kind="probability",
        ),
        feature_state=FeatureState(engineered_test_path=str(engineered_test)),
        model_state=ModelState(
            candidate_models=["ModelA"],
            best_model_name="ModelA",
            validation_metrics={"score": 1.0},
            cv_results={"ModelA": {"score": 1.0}},
            prediction_path=str(prediction_file),
            baseline_metrics={"score": 1.0},
        ),
        review_state=ReviewState(),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "model_results": str(model_results),
            "best_model": str(best_model),
            "prediction_file": str(prediction_file),
            "phase_plan_records": {"model_building": str(model_plan)},
        },
    )

    result = asyncio.run(ReviewerAgent().run(state))

    assert result["review_state"]["latest_review_status"] == "approved"
