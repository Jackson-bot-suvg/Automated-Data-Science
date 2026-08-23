from app.schemas.artifact_schema import DEFAULT_ARTIFACT_PATHS
from app.schemas.state_schema import GraphState


def test_graph_state_smoke() -> None:
    state = GraphState()

    assert state.task_config.task_id == ""
    assert state.current_phase == ""
    assert isinstance(state.artifacts, dict)
    assert state.artifacts["preliminary_eda_report"] == DEFAULT_ARTIFACT_PATHS["preliminary_eda_report"]
    assert state.artifacts["figures"] == []

    required_keys = {
        "preliminary_eda_report",
        "cleaned_train_data",
        "cleaned_test_data",
        "cleaning_report",
        "indepth_eda_report",
        "feature_summary",
        "engineered_train_data",
        "engineered_test_data",
        "model_results",
        "best_model",
        "prediction_file",
        "technical_report",
        "business_report",
        "run_manifest",
        "phase_plan_records",
        "phase_review_records",
        "figures",
    }

    assert required_keys.issubset(DEFAULT_ARTIFACT_PATHS.keys())
    assert state.task_config.constraints.interpretability_required is True
    assert state.problem_understanding.data_overview.n_rows == 0
    assert state.runtime_state.thread_id == ""
    assert state.runtime_state.checkpoint_namespace == "default"
    assert state.runtime_state.interrupted is False


def test_eda_partial_update_merged_without_schema_error() -> None:
    """模拟 run_preliminary_eda 的返回值能与 GraphState 无冲突地合并。"""
    from app.schemas.state_schema import DataState
    state = GraphState()
    simulated_return = {
        "data_state": state.data_state.model_dump()
        | {
            "schema_summary": {"columns": ["Age"]},
            "eda_findings": {"dataset_shape": {"rows": 1}, "columns": {"Age": {"type": "numeric"}}},
        },
        "artifacts": {**state.artifacts, "preliminary_eda_report": "artifacts/eda/preliminary_eda.md"},
        "current_phase": "preliminary_eda",
        "phase_history": ["preliminary_eda"],
    }
    new_data_state = DataState(**simulated_return["data_state"])
    assert new_data_state.eda_findings["columns"]["Age"]["type"] == "numeric"
    assert simulated_return["artifacts"]["preliminary_eda_report"] == "artifacts/eda/preliminary_eda.md"


def test_cleaning_partial_update_merged_without_schema_error() -> None:
    """模拟 run_data_cleaning 的返回值能与 GraphState 无冲突地合并。"""
    from app.schemas.state_schema import DataState
    state = GraphState()
    simulated_return = {
        "data_state": state.data_state.model_dump()
        | {
            "current_train_path": "artifacts/data/cleaned_train.csv",
            "current_test_path": "artifacts/data/cleaned_test.csv",
            "cleaning_summary": {"age_imputation": {"strategy": "分组中位数"}},
            "data_quality_flags": ["Cabin 已删除"],
        },
        "artifacts": {
            **state.artifacts,
            "cleaned_train_data": "artifacts/data/cleaned_train.csv",
            "cleaned_test_data": "artifacts/data/cleaned_test.csv",
            "cleaning_report": "artifacts/data/cleaning_report.md",
        },
        "current_phase": "data_cleaning",
        "phase_history": ["data_cleaning"],
    }
    new_data_state = DataState(**simulated_return["data_state"])
    assert new_data_state.current_train_path == "artifacts/data/cleaned_train.csv"
    assert "Cabin 已删除" in new_data_state.data_quality_flags


def test_feature_partial_update_merged_without_schema_error() -> None:
    """模拟 run_feature_engineering 的返回值能与 GraphState 无冲突地合并。"""
    from app.schemas.state_schema import FeatureState
    state = GraphState()
    simulated_return = {
        "feature_state": state.feature_state.model_dump()
        | {
            "engineered_train_path": "artifacts/features/engineered_train.csv",
            "engineered_test_path": "artifacts/features/engineered_test.csv",
            "selected_features": ["Pclass", "Sex", "Title", "FamilySize", "IsAlone", "FareBand", "AgeBand"],
            "dropped_features": ["Name", "SibSp", "Parch", "Age", "Fare"],
        },
        "artifacts": {**state.artifacts, "feature_summary": "artifacts/features/feature_summary.json"},
        "current_phase": "feature_engineering",
        "phase_history": ["feature_engineering"],
    }
    new_feature_state = FeatureState(**simulated_return["feature_state"])
    assert "Title" in new_feature_state.selected_features
    assert "Name" in new_feature_state.dropped_features
