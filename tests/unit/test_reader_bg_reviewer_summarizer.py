from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

from app.agents.bg_reviewer import run_bg_reviewer
from app.agents.reader import run_reader
from app.agents.summarizer import TechnicalSummarizerAgent, run_technical_summary
from app.graph.routes import route_after_reviewer, route_after_summarizer
from app.schemas.artifact_schema import DEFAULT_ARTIFACT_PATHS
from app.schemas.state_schema import (
    DataOverview,
    DataState,
    EvaluationSpec,
    FeatureState,
    GraphState,
    ModelState,
    ProblemUnderstanding,
    ReviewState,
    RuntimeState,
    SubmissionSpec,
    TaskConfig,
    TargetSpec,
)


def _write_titanic_like_data(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 1],
            "Age": [22.0, None, 35.0],
            "Fare": [7.25, 71.28, 8.05],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {
            "PassengerId": [4, 5],
            "Age": [28.0, 40.0],
            "Fare": [13.0, 9.5],
        }
    ).to_csv(test_path, index=False)
    return train_path, test_path


def _write_churn_like_directory(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "bank_churn"
    dataset_dir.mkdir()
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "CreditScore": [700, 620, 590],
            "Age": [42, 35, 28],
            "Exited": [1, 0, 0],
        }
    ).to_csv(dataset_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [4, 5],
            "CreditScore": [680, 610],
            "Age": [31, 49],
        }
    ).to_csv(dataset_dir / "test.csv", index=False)
    pd.DataFrame({"id": [4, 5], "Exited": [0.5, 0.5]}).to_csv(
        dataset_dir / "sample_submission.csv", index=False
    )
    (dataset_dir / "overview.txt").write_text(
        "Your goal is to predict whether a customer churns. "
        "The target variable `Exited` is binary.",
        encoding="utf-8",
    )
    return dataset_dir


def _write_plate_defect_like_directory(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "plate_defect"
    dataset_dir.mkdir()
    target_columns = [f"Defect_{index}" for index in range(1, 8)]
    pd.DataFrame(
        {
            "ImageId": [1, 2, 3],
            "Steel_Plate_Thickness": [1.2, 0.9, 1.5],
            "Surface_Roughness": [0.3, 0.5, 0.4],
            **{column: [0, 1, 0] for column in target_columns},
        }
    ).to_csv(dataset_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "ImageId": [4, 5],
            "Steel_Plate_Thickness": [1.1, 1.0],
            "Surface_Roughness": [0.2, 0.6],
        }
    ).to_csv(dataset_dir / "test.csv", index=False)
    pd.DataFrame(
        {
            "ImageId": [4, 5],
            **{column: [0.1, 0.2] for column in target_columns},
        }
    ).to_csv(dataset_dir / "sample_submission.csv", index=False)
    (dataset_dir / "overview.txt").write_text(
        "Predict the defect labels for each steel plate image.",
        encoding="utf-8",
    )
    return dataset_dir


@pytest.mark.asyncio
async def test_reader_initializes_problem_and_data_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("READER_LLM_ENABLED", "false")
    train_path, test_path = _write_titanic_like_data(tmp_path)
    state = GraphState(
        task_config=TaskConfig(
            task_id="titanic-mvp",
            target_column="Survived",
            business_context="Predict passenger survival.",
            dataset_paths={
                "train": str(train_path),
                "test": str(test_path),
            },
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_reader(state)

    assert result["current_phase"] == "reader"
    assert result["phase_history"] == ["reader"]
    assert result["target_spec"]["status"] == "resolved"
    assert result["target_spec"]["target_mode"] == "single_target"
    assert result["target_spec"]["target_columns"] == ["Survived"]
    assert result["problem_understanding"]["target_column"] == ""
    assert result["problem_understanding"]["target_columns"] == []
    assert result["task_config"]["target_column"] == ""
    assert result["task_config"]["target_columns"] == []
    assert result["data_state"]["target_column"] == ""
    assert result["data_state"]["target_columns"] == []
    assert result["problem_understanding"]["predicted_task_type"] == "binary_classification"
    assert "roc_auc" in result["problem_understanding"]["candidate_metrics"]
    assert result["data_state"]["current_train_path"] == str(train_path.resolve())
    assert result["data_state"]["current_test_path"] == str(test_path.resolve())
    assert result["data_state"]["train_shape"] == (3, 4)


@pytest.mark.asyncio
async def test_reader_records_missing_path_risk(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("READER_LLM_ENABLED", "false")
    missing_train = tmp_path / "missing.csv"
    state = GraphState(
        task_config=TaskConfig(
            dataset_paths={"train": str(missing_train)},
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_reader(state)

    assert result["current_phase"] == "reader"
    assert result["data_state"]["current_train_path"] == str(missing_train.resolve())
    assert any(
        "not readable" in risk for risk in result["problem_understanding"]["risks"]
    )


@pytest.mark.asyncio
async def test_reader_discovers_kaggle_directory_and_submission_target(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("READER_LLM_ENABLED", "false")
    dataset_dir = _write_churn_like_directory(tmp_path)
    state = GraphState(
        task_config=TaskConfig(
            raw_task="用这个数据集预测客户流失",
            dataset_root=str(dataset_dir),
            business_context="用这个数据集预测客户流失",
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_reader(state)

    assert result["task_config"]["dataset_paths"]["train"].endswith("train.csv")
    assert result["task_config"]["dataset_paths"]["test"].endswith("test.csv")
    assert result["task_config"]["dataset_paths"]["submission"].endswith(
        "sample_submission.csv"
    )
    assert result["target_spec"]["status"] == "resolved"
    assert result["target_spec"]["target_mode"] == "single_target"
    assert result["target_spec"]["submission"]["id_columns"] == ["id"]
    assert result["target_spec"]["submission"]["output_columns"] == ["Exited"]
    assert result["target_spec"]["prediction_kind"] == "probability"
    assert result["problem_understanding"]["target_column"] == ""
    assert result["problem_understanding"]["target_columns"] == []
    assert "roc_auc" in result["problem_understanding"]["candidate_metrics"]
    assert result["data_state"]["schema_summary"]["overview_files"]


@pytest.mark.asyncio
async def test_reader_prefers_structured_multi_target_scope(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("READER_LLM_ENABLED", "false")
    dataset_dir = _write_plate_defect_like_directory(tmp_path)
    state = GraphState(
        task_config=TaskConfig(
            raw_task="预测钢板缺陷标签，不要只看厚度",
            dataset_root=str(dataset_dir),
            business_context="预测钢板缺陷标签，不要只看厚度",
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_reader(state)

    assert result["target_spec"]["status"] == "resolved"
    assert result["target_spec"]["target_mode"] == "multi_label"
    assert result["target_spec"]["submission"]["output_columns"] == [
        "Defect_1",
        "Defect_2",
        "Defect_3",
        "Defect_4",
        "Defect_5",
        "Defect_6",
        "Defect_7",
    ]
    assert result["problem_understanding"]["target_column"] == ""
    assert result["problem_understanding"]["target_columns"] == []
    assert "7 target columns" in result["problem_understanding"]["task_summary"]


@pytest.mark.asyncio
async def test_reader_warns_when_test_csv_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("READER_LLM_ENABLED", "false")
    dataset_dir = tmp_path / "train_only"
    dataset_dir.mkdir()
    pd.DataFrame({"target": [0, 1], "x": [1, 2]}).to_csv(
        dataset_dir / "train.csv", index=False
    )

    result = await run_reader(
        GraphState(
            task_config=TaskConfig(
                raw_task="predict target",
                dataset_root=str(dataset_dir),
                target_hint="target",
            ),
            runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        )
    )

    assert result["data_state"]["current_test_path"] == ""
    assert any("No external test.csv" in risk for risk in result["problem_understanding"]["risks"])


@pytest.mark.asyncio
async def test_reader_uses_natural_language_target_hint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("READER_LLM_ENABLED", "false")
    dataset_dir = _write_churn_like_directory(tmp_path)
    state = GraphState(
        task_config=TaskConfig(
            raw_task="分析银行客户",
            dataset_root=str(dataset_dir),
            target_hint="我要预测客户是否流失",
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_reader(state)

    assert result["problem_understanding"]["target_column"] == ""
    assert result["target_spec"]["status"] == "resolved"
    assert result["target_spec"]["user_target_hint"] == "我要预测客户是否流失"


@pytest.mark.asyncio
async def test_reader_leaves_target_empty_when_it_cannot_infer(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("READER_LLM_ENABLED", "false")
    dataset_dir = tmp_path / "ambiguous"
    dataset_dir.mkdir()
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(
        dataset_dir / "train.csv", index=False
    )

    result = await run_reader(
        GraphState(
            task_config=TaskConfig(raw_task="predict something", dataset_root=str(dataset_dir)),
            runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        )
    )

    assert result["problem_understanding"]["target_column"] == ""
    assert result["problem_understanding"]["target_columns"] == []
    assert result["target_spec"]["status"] == "unresolved"
    assert result["target_spec"]["target_columns"] == []
    assert result["target_spec"]["evidence"][0]["source"] == "unresolved"


@pytest.mark.asyncio
async def test_background_reviewer_approves_reader_output(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BG_REVIEWER_LLM_ENABLED", "false")
    train_path, test_path = _write_titanic_like_data(tmp_path)
    state = _reader_ready_state(tmp_path, train_path, test_path)

    result = await run_bg_reviewer(state)

    assert result["review_state"]["latest_review_status"] == "approved"
    assert result["review_state"]["pending_review_phase"] == "reader"
    assert result["runtime_state"]["last_reviewed_phase"] == "reader"
    assert result["runtime_state"]["interrupted"] is False
    assert result["review_state"]["review_history"][0]["phase"] == "reader"


@pytest.mark.asyncio
async def test_background_reviewer_retries_missing_target(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BG_REVIEWER_LLM_ENABLED", "false")
    train_path, test_path = _write_titanic_like_data(tmp_path)
    state = _reader_ready_state(tmp_path, train_path, test_path)
    state.target_spec = TargetSpec()

    result = await run_bg_reviewer(state)

    assert result["review_state"]["latest_review_status"] == "retry"
    assert result["review_state"]["retry_count_by_phase"]["reader"] == 1
    assert result["runtime_state"]["interrupted"] is False
    assert "target unresolved" in result["review_state"]["review_history"][0]["blocking_failures"][0]


@pytest.mark.asyncio
async def test_background_reviewer_fatal_for_missing_train_file(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BG_REVIEWER_LLM_ENABLED", "false")
    missing_train = tmp_path / "missing.csv"
    state = GraphState(
        problem_understanding=ProblemUnderstanding(
            task_summary="classification task",
            target_column="Survived",
            candidate_metrics=["accuracy"],
            data_overview=DataOverview(column_types={"Survived": "int64"}),
        ),
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["Survived"],
            target_mode="single",
            task_family="classification",
            prediction_kind="label",
            evaluation=EvaluationSpec(candidate_metrics=["accuracy"]),
        ),
        data_state=DataState(
            current_train_path=str(missing_train), target_column="Survived"
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_bg_reviewer(state)

    assert result["review_state"]["latest_review_status"] == "fatal"
    assert result["runtime_state"]["interrupted"] is True
    assert result["review_state"]["fatal_errors"]


@pytest.mark.asyncio
async def test_background_reviewer_allows_missing_test_with_warning(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BG_REVIEWER_LLM_ENABLED", "false")
    train_path, _ = _write_titanic_like_data(tmp_path)
    state = _reader_ready_state(tmp_path, train_path, Path(""))
    state.data_state.current_test_path = ""

    result = await run_bg_reviewer(state)

    assert result["review_state"]["latest_review_status"] == "approved"
    warnings = result["review_state"]["review_history"][0]["warnings"]
    assert any("train-only validation" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_background_reviewer_approves_multi_target_scope(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BG_REVIEWER_LLM_ENABLED", "false")
    dataset_dir = _write_plate_defect_like_directory(tmp_path)
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    target_columns = [f"Defect_{index}" for index in range(1, 8)]
    state = GraphState(
        task_config=TaskConfig(target_columns=target_columns),
        target_spec=TargetSpec(
            status="resolved",
            target_columns=target_columns,
            target_mode="multi",
            task_family="classification",
            prediction_kind="probability",
            submission=SubmissionSpec(
                id_columns=["ImageId"],
                output_columns=target_columns,
                submission_columns=["ImageId", *target_columns],
                sample_submission_path=str(dataset_dir / "sample_submission.csv"),
            ),
            evaluation=EvaluationSpec(candidate_metrics=["f1", "roc_auc"]),
        ),
        problem_understanding=ProblemUnderstanding(
            task_summary="classification task for plate defects",
            predicted_task_type="classification",
            target_columns=target_columns,
            candidate_metrics=["f1", "roc_auc"],
            data_overview=DataOverview(
                n_rows=3,
                n_cols=10,
                column_types={column: "int64" for column in target_columns},
            ),
        ),
        data_state=DataState(
            current_train_path=str(train_path),
            current_test_path=str(test_path),
            target_columns=target_columns,
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_bg_reviewer(state)

    assert result["review_state"]["latest_review_status"] == "approved"


@pytest.mark.asyncio
async def test_background_reviewer_retries_when_test_contains_target_columns(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BG_REVIEWER_LLM_ENABLED", "false")
    train_path, test_path = _write_titanic_like_data(tmp_path)
    pd.DataFrame(
        {
            "PassengerId": [4, 5],
            "Survived": [0, 1],
            "Age": [28.0, 40.0],
            "Fare": [13.0, 9.5],
        }
    ).to_csv(test_path, index=False)
    state = _reader_ready_state(tmp_path, train_path, test_path)

    result = await run_bg_reviewer(state)

    assert result["review_state"]["latest_review_status"] == "retry"
    assert "target leakage" in result["review_state"]["review_history"][0]["blocking_failures"][0]


@pytest.mark.asyncio
async def test_background_reviewer_retries_on_submission_schema_mismatch(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BG_REVIEWER_LLM_ENABLED", "false")
    dataset_dir = _write_churn_like_directory(tmp_path)
    wrong_submission = dataset_dir / "sample_submission_wrong.csv"
    pd.DataFrame({"Exited": [0.5, 0.5], "id": [4, 5]}).to_csv(
        wrong_submission, index=False
    )
    state = GraphState(
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["Exited"],
            target_mode="single",
            task_family="classification",
            prediction_kind="probability",
            submission=SubmissionSpec(
                id_columns=["id"],
                output_columns=["Exited"],
                submission_columns=["id", "Exited"],
                sample_submission_path=str(wrong_submission),
            ),
            evaluation=EvaluationSpec(candidate_metrics=["roc_auc"]),
        ),
        problem_understanding=ProblemUnderstanding(
            task_summary="classification task for churn",
            predicted_task_type="classification",
            target_column="Exited",
            target_columns=["Exited"],
            candidate_metrics=["roc_auc"],
            data_overview=DataOverview(column_types={"Exited": "int64"}),
        ),
        data_state=DataState(
            current_train_path=str(dataset_dir / "train.csv"),
            current_test_path=str(dataset_dir / "test.csv"),
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_bg_reviewer(state)

    assert result["review_state"]["latest_review_status"] == "retry"
    assert "submission mismatch" in result["review_state"]["review_history"][0]["blocking_failures"][0]


@pytest.mark.asyncio
async def test_technical_summarizer_noops_before_final_phase(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SUMMARIZER_LLM_ENABLED", "false")
    state = GraphState(
        current_phase="preliminary_eda",
        review_state=ReviewState(latest_review_status="approved"),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )

    result = await run_technical_summary(state)

    assert "report_state" not in result
    assert "current_phase" not in result
    assert route_after_reviewer(state) == "approved"
    assert route_after_summarizer(state) == "next_phase"


@pytest.mark.asyncio
async def test_technical_summarizer_generates_final_report(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SUMMARIZER_LLM_ENABLED", "false")
    state = GraphState(
        current_phase="model_building",
        task_config=TaskConfig(
            target_column="Survived", business_context="Survival decision support."
        ),
        problem_understanding=ProblemUnderstanding(
            task_summary="Predict survival",
            predicted_task_type="classification",
            target_column="Survived",
            candidate_metrics=["accuracy"],
            data_overview=DataOverview(
                n_rows=3, n_cols=4, column_types={"Survived": "int64"}
            ),
        ),
        data_state=DataState(
            schema_summary={"columns": ["PassengerId", "Survived", "Age", "Fare"]},
            eda_findings={"dataset_shape": {"rows": 3, "columns": 4}},
            cleaning_summary={"fill_missing": {"columns": ["Age"]}},
        ),
        feature_state=FeatureState(
            selected_features=["Age", "Fare"],
            feature_transform_summary={"selected_features": ["Age", "Fare"]},
        ),
        model_state=ModelState(
            candidate_models=["LogisticRegression"],
            best_model_name="LogisticRegression",
            validation_metrics={"accuracy": 0.8},
            prediction_path=str(tmp_path / "submission.csv"),
        ),
        review_state=ReviewState(
            latest_review_status="approved",
            review_history=[
                {
                    "phase": "model_building",
                    "status": "approved",
                    "attempt": 1,
                    "comment": "model approved",
                }
            ],
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        artifacts={
            **DEFAULT_ARTIFACT_PATHS,
            "prediction_file": str(tmp_path / "submission.csv"),
        },
    )

    result = await run_technical_summary(state)

    assert result["current_phase"] == "technical_summary"
    assert result["phase_history"] == ["technical_summary"]
    technical_report_path = Path(result["report_state"]["technical_report_path"])
    assert technical_report_path.exists()
    assert technical_report_path.as_posix().endswith(
        DEFAULT_ARTIFACT_PATHS["technical_report"]
    )
    report_text = technical_report_path.read_text(encoding="utf-8")
    assert "# Technical Report" in report_text
    assert "## Executive Summary" in report_text
    assert "## Modeling Summary" in report_text
    assert result["artifacts"]["technical_report"] == str(technical_report_path)
    assert (
        route_after_summarizer(GraphState(current_phase=result["current_phase"]))
        == "final"
    )


def test_route_after_phase_reviewer_only_summarizes_after_model_building() -> None:
    assert (
        route_after_reviewer(
            GraphState(
                current_phase="feature_engineering",
                review_state=ReviewState(latest_review_status="approved"),
            )
        )
        == "approved"
    )
    assert (
        route_after_reviewer(
            GraphState(
                current_phase="model_building",
                review_state=ReviewState(latest_review_status="approved"),
            )
        )
        == "summarizer"
    )


def test_technical_summarizer_prefers_target_spec_contract_in_report() -> None:
    agent = TechnicalSummarizerAgent()
    state = GraphState(
        current_phase="model_building",
        task_config=TaskConfig(
            business_context="识别钢板缺陷，辅助质检团队优先复检高风险样本。"
        ),
        problem_understanding=ProblemUnderstanding(
            task_summary="预测每块钢板是否命中多个缺陷标签",
            predicted_task_type="classification",
        ),
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["Pastry", "Z_Scratch", "K_Scatch"],
            target_mode="multi_label",
            task_family="multilabel_classification",
            prediction_kind="probability",
            submission=SubmissionSpec(
                id_columns=["ImageId"],
                output_columns=["Pastry", "Z_Scratch", "K_Scatch"],
                submission_columns=["ImageId", "Pastry", "Z_Scratch", "K_Scatch"],
                sample_submission_path="data/plate/sample_submission.csv",
            ),
            evaluation=EvaluationSpec(
                primary_metric="f1_micro",
                candidate_metrics=["f1_micro", "f1_macro", "roc_auc_macro_ovr"],
            ),
        ),
        data_state=DataState(
            train_shape=(100, 12),
            test_shape=(20, 9),
            cleaning_summary={"missing_value_strategy": "median"},
        ),
        feature_state=FeatureState(
            selected_features=["Thickness", "Roughness", "X_Minimum"],
            leakage_checks=["PASS: 目标列已从 selected_features 中排除"],
            feature_importance_summary={"Thickness": 0.51, "Roughness": 0.33},
        ),
        model_state=ModelState(
            candidate_models=["OneVsRestClassifier", "RandomForestClassifier"],
            best_model_name="OneVsRestClassifier",
            validation_metrics={
                "f1_micro": 0.8123,
                "f1_macro": 0.7744,
                "_meta": {
                    "task_family": "multilabel_classification",
                    "target_mode": "multi_label",
                    "primary_metric": "f1_micro",
                    "prediction_kind": "probability",
                },
            },
            prediction_path="artifacts/predictions/submission.csv",
        ),
        review_state=ReviewState(
            latest_review_status="approved",
            review_history=[
                {
                    "phase": "model_building",
                    "status": "approved",
                    "attempt": 1,
                    "comment": "<think>ignore</think> metrics and submission look good",
                }
            ],
        ),
    )

    summary_context = agent._summary_context(state)
    payload = agent._build_summary_payload(summary_context)
    report = payload["report_markdown"]

    assert "Pastry" in report
    assert "Z_Scratch" in report
    assert "multilabel_classification" in report
    assert "f1_micro" in report
    assert "sample_submission.csv" in report
    assert "<think>" not in report
    assert "## Business Translation Inputs" in report
    assert not re.search(r"[\u3400-\u9fff]", report)


def _reader_ready_state(
    tmp_path: Path, train_path: Path, test_path: Path
) -> GraphState:
    return GraphState(
        task_config=TaskConfig(target_column="Survived"),
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["Survived"],
            target_mode="single",
            task_family="classification",
            prediction_kind="label",
            submission=SubmissionSpec(),
            evaluation=EvaluationSpec(candidate_metrics=["accuracy"]),
        ),
        problem_understanding=ProblemUnderstanding(
            task_summary="classification task",
            predicted_task_type="classification",
            target_column="Survived",
            candidate_metrics=["accuracy"],
            data_overview=DataOverview(
                n_rows=3,
                n_cols=4,
                column_types={"Survived": "int64", "Age": "float64"},
            ),
        ),
        data_state=DataState(
            current_train_path=str(train_path),
            current_test_path=str(test_path),
            target_column="Survived",
        ),
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
    )
