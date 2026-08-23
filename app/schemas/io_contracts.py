from __future__ import annotations

from typing import Final

REVIEW_GATED_PHASES: Final[tuple[str, ...]] = (
    "preliminary_eda",
    "data_cleaning",
    "indepth_eda", 
    "feature_engineering", 
    "model_building",
    "technical_summary",
    "business_translation",
)


PHASE_SEQUENCE: Final[tuple[str, ...]] = (
    "reader",
    "planner",
    "preliminary_eda",
    "data_cleaning",
    "indepth_eda",
    "feature_engineering",
    "model_building",
    "technical_summary",
    "business_translation",
)

LANGGRAPH_RUNTIME_CONTRACT: Final[dict[str, str]] = {
    "builder": "StateGraph(GraphState)",
    "entrypoint": "START -> reader",
    "terminal": "business_translation -> END",
    "routing": "Use add_conditional_edges(...) or Command(...) for review gates and retries.",
    "persistence": "Compile the graph with a checkpointer and invoke it with configurable.thread_id.",
    "interrupts": "Human approval and reviewer gates must pause and resume through interrupt/resume semantics.",
}

STATE_REDUCER_RULES: Final[dict[str, str]] = {
    "phase_history": "append_only",
    "review_state.review_history": "append_only",
    "artifacts.figures": "append_only",
    "messages": "append_only_messages",
}

NODE_CONTRACTS: Final[dict[str, dict[str, object]]] = {
    "reader": {
        "entry": "run_reader(state: GraphState) -> dict",
        "reads": ["task_config"],
        "writes": [
            "problem_understanding",
            "target_spec",
            "data_state.raw_train_path",
            "data_state.raw_test_path",
            "data_state.current_train_path",
            "data_state.current_test_path",
            "current_phase",
            "phase_history",
        ],
    },
    "bg_reviewer": {
        "entry": "run_bg_reviewer(state: GraphState) -> dict",
        "reads": ["problem_understanding", "current_phase", "phase_history"],
        "writes": [
            "review_state.latest_review_status",
            "review_state.last_review_comment",
            "review_state.review_history",
            "review_state.pending_review_phase",
            "runtime_state.last_reviewed_phase",
            "runtime_state.interrupted",
        ],
    },
    "planner": {
        "entry": "run_planner(state: GraphState) -> dict",
        "reads": [
            "task_config",
            "problem_understanding",
            "target_spec",
            "review_state",
            "artifacts",
        ],
        "writes": ["execution_plan"],
    },
    "preliminary_eda": {
        "entry": "run_preliminary_eda(state: GraphState) -> dict",
        "reads": ["data_state.current_train_path", "problem_understanding.target_column"],
        "writes": [
            "data_state.schema_summary",
            "data_state.eda_findings",
            "artifacts.preliminary_eda_report",
            "artifacts.figures",
        ],
    },
    "data_cleaning": {
        "entry": "run_data_cleaning(state: GraphState) -> dict",
        "reads": [
            "data_state.current_train_path",
            "data_state.current_test_path",
            "data_state.eda_findings",
            "target_spec",
            "execution_plan.success_criteria.data_cleaning.agent_plan",
        ],
        "writes": [
            "data_state.current_train_path",
            "data_state.current_test_path",
            "data_state.cleaning_summary",
            "artifacts.cleaned_train_data",
            "artifacts.cleaned_test_data",
            "artifacts.cleaning_report",
            "artifacts.phase_plan_records.data_cleaning",
        ],
    },

    "indepth_eda": {
        "entry": "run_indepth_eda(state: GraphState) -> dict",     
        "reads": [
            "data_state.current_train_path",
            "data_state.eda_findings",
            "data_state.schema_summary",
            "target_spec",
            "execution_plan.success_criteria.indepth_eda.agent_plan",
        ],
        "writes": [
            "data_state.indepth_eda_summary",
            "artifacts.indepth_eda_report",
            "artifacts.figures",
            "artifacts.phase_plan_records.indepth_eda",
        ],
    },

    "feature_engineering": {
        "entry": "run_feature_engineering(state: GraphState) -> dict",
        "reads": [
            "data_state.current_train_path",
            "data_state.current_test_path",
            "target_spec",
            "execution_plan.success_criteria.feature_engineering.agent_plan",
        ],
        "writes": [
            "feature_state.engineered_train_path",
            "feature_state.engineered_test_path",
            "feature_state.selected_features",
            "feature_state.feature_transform_summary",
            "artifacts.feature_summary",
            "artifacts.phase_plan_records.feature_engineering",
        ],
    },
    "model_building": {
        "entry": "run_model_building(state: GraphState) -> dict",
        "reads": [
            "feature_state.engineered_train_path",
            "feature_state.engineered_test_path",
            "feature_state.selected_features",
            "target_spec",
            "execution_plan.success_criteria.model_building.agent_plan",
        ],
        "writes": [
            "model_state.candidate_models",
            "model_state.best_model_name",
            "model_state.validation_metrics",
            "model_state.prediction_path",
            "model_state.model_explanations",
            "feature_state.feature_importance_summary",
            "artifacts.model_results",
            "artifacts.best_model",
            "artifacts.prediction_file",
            "artifacts.phase_plan_records.model_building",
        ],
    },
    "reviewer": {
        "entry": "run_reviewer(state: GraphState) -> dict",
        "reads": ["current_phase", "execution_plan.success_criteria", "artifacts"],
        "writes": [
            "review_state.latest_review_status",
            "review_state.last_review_comment",
            "review_state.review_history",
            "review_state.pending_review_phase",
            "runtime_state.last_reviewed_phase",
            "runtime_state.interrupted",
            "artifacts.phase_review_records",
        ],
    },
    "technical_summary": {
        "entry": "run_technical_summary(state: GraphState) -> dict",
        "reads": ["data_state", "feature_state", "model_state", "review_state"],
        "writes": ["report_state.technical_report_path", "artifacts.technical_report"],
    },
    "business_translation": {
        "entry": "run_business_translation(state: GraphState) -> dict",
        "reads": [
            "report_state.technical_report_path",
            "model_state.validation_metrics",
            "feature_state.feature_importance_summary",
            "task_config.business_context",
        ],
        "writes": [
            "report_state.business_report_path",
            "report_state.executive_summary",
            "report_state.recommended_actions",
            "artifacts.business_report",
        ],
    },
}
