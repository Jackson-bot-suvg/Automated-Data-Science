from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from app.agents.business_translator import (
    _compact_artifact_evidence_for_llm,
    _infer_industry_profile,
    _render_business_report_markdown,
    run_business_translation,
)
from app.schemas.artifact_schema import DEFAULT_ARTIFACT_PATHS
from app.schemas.state_schema import GraphState


def test_business_translation_generates_report_and_state_updates() -> None:
    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        technical_report_path = base_dir / DEFAULT_ARTIFACT_PATHS["technical_report"]
        technical_report_path.parent.mkdir(parents=True, exist_ok=True)
        technical_report_path.write_text("# Technical Report\n\nBest model: demo\n", encoding="utf-8")

        state = GraphState(
            task_config={"business_context": "Support retention decisions for subscription users."},
            problem_understanding={"predicted_task_type": "classification"},
            model_state={"validation_metrics": {"accuracy": 0.81, "f1": 0.78}},
            feature_state={"feature_importance_summary": {"tenure_days": 0.6, "support_tickets": 0.4}},
            report_state={"technical_report_path": str(technical_report_path)},
            runtime_state={"workspace_root": str(base_dir)},
        )

        result = run_business_translation(state)

        assert result["current_phase"] == "business_translation"
        assert "report_state" in result
        assert "artifacts" in result

        business_report_path = Path(result["report_state"]["business_report_path"])
        assert business_report_path.exists()
        assert business_report_path.as_posix().endswith(DEFAULT_ARTIFACT_PATHS["business_report"])

        assert result["artifacts"]["business_report"] == str(business_report_path)
        assert isinstance(result["report_state"]["recommended_actions"], list)
        assert result["report_state"]["executive_summary"]


def test_infer_industry_profile_uses_artifact_columns_signal() -> None:
    artifact_evidence = {
        "artifact_paths": ["artifacts/intermediate/orders.csv"],
        "artifact_summaries": {
            "artifacts/intermediate/orders.csv": {
                "type": "csv",
                "rows": 1200,
                "columns": ["order_id", "cart_value", "conversion_rate", "customer_id"],
            }
        },
    }
    profile = _infer_industry_profile(
        dataset_name="daily_orders",
        business_context="improve checkout conversion and basket size",
        technical_report_text="",
        artifact_evidence=artifact_evidence,
    )
    assert profile
    assert profile["industry_name"] == "E-commerce / Digital Retail"


def test_compact_artifact_evidence_for_llm_removes_heavy_payloads() -> None:
    artifact_evidence = {
        "artifact_paths": [f"artifacts/file_{idx}.csv" for idx in range(8)],
        "artifact_summaries": {
            "artifacts/file_1.csv": {
                "type": "csv",
                "rows": 100,
                "columns": [f"col_{idx}" for idx in range(20)],
                "sample_rows": [["a", "b"], ["c", "d"]],
            },
            "artifacts/file_2.md": {
                "type": "markdown",
                "summary": "A" * 800,
            },
        },
    }
    compact = _compact_artifact_evidence_for_llm(artifact_evidence)
    assert len(compact["artifact_paths"]) <= 4
    csv_summary = compact["artifact_summaries"]["artifacts/file_1.csv"]
    assert "sample_rows" not in csv_summary
    assert len(csv_summary["columns"]) <= 10


def test_infer_industry_profile_identifies_passenger_transport_context() -> None:
    artifact_evidence = {
        "artifact_paths": ["artifacts/intermediate/cleaned_train.csv"],
        "artifact_summaries": {
            "artifacts/intermediate/cleaned_train.csv": {
                "type": "csv",
                "rows": 891,
                "columns": ["Pclass", "Sex", "Age", "Fare", "Embarked", "Cabin", "Survived"],
            }
        },
    }
    profile = _infer_industry_profile(
        dataset_name="passenger_survival",
        business_context="optimize voyage safety decisions",
        technical_report_text="Passenger survival model for voyage operations.",
        artifact_evidence=artifact_evidence,
    )
    assert profile
    assert profile["industry_name"] == "Premium Cross-Border Passenger Transport & Maritime Tourism"


def test_industry_profile_covers_all_repo_datasets() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    dataset_dirs = [path for path in data_root.iterdir() if path.is_dir()]
    assert dataset_dirs

    for dataset_dir in dataset_dirs:
        profile = _infer_industry_profile(
            dataset_name=dataset_dir.name,
            business_context="",
            technical_report_text="",
            artifact_evidence={},
        )
        assert profile
        assert profile.get("industry_name")


def test_rendered_report_uses_required_business_sections() -> None:
    report = _render_business_report_markdown(
        executive_summary="Model evidence supports targeted segmentation decisions.",
        business_context="Improve customer retention and service efficiency.",
        task_family="multiclass_classification",
        dataset_name="demo_dataset",
        validation_metrics={"accuracy": 0.81},
        baseline_metrics={"accuracy": 0.72},
        top_drivers=[],
        limitations=["Sample size is limited."],
        recommended_actions=["P0: Launch segmented interventions with weekly KPI review."],
        execution_suggestions=["Owner: ops + data team."],
        insight_points=["Risk segmentation outperforms baseline and is suitable for pilot rollout."],
        chart_guidance=["Metric charts explain pilot upside and downside."],
        chart_paths={},
        technical_report_excerpt="best model: demo",
        artifact_evidence={},
        insight_briefs=[],
        action_briefs=[],
        domain_context={"industry_name": "General Predictive Operations"},
    )
    for heading in (
        "## Preface",
        "## Business Context Modeling",
        "## Core Insight Discovery",
        "## Strategic Action Recommendations",
        "## Business Brief Summary",
    ):
        assert heading in report
