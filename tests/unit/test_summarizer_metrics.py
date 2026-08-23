from app.agents.summarizer import TechnicalSummarizerAgent


def test_executive_summary_uses_best_model_metric_mapping() -> None:
    agent = TechnicalSummarizerAgent()
    summary_context = {
        "target_contract": {
            "primary_metric": "f1_macro",
            "task_family": "multiclass_classification",
            "target_columns": ["type"],
        },
        "model_state": {
            "best_model_name": "LogisticRegression",
            "validation_metrics": {
                "SVC": {"cv_f1_macro_mean": 0.7284},
                "LogisticRegression": {"cv_f1_macro_mean": 0.7639},
            },
        },
    }

    summary = agent._build_executive_summary(summary_context)

    assert "LogisticRegression" in summary
    assert "f1_macro=0.7639" in summary
    assert "0.7284" not in summary
