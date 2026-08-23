from __future__ import annotations

import operator
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.artifact_schema import DEFAULT_ARTIFACT_PATHS


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetPaths(StrictBaseModel):
    train: str = ""
    test: str = ""
    submission: str = ""


class TaskConstraints(StrictBaseModel):
    max_runtime_min: int | None = None
    max_tokens: int | None = None
    interpretability_required: bool = True
    business_translation_required: bool = True


class TaskConfig(StrictBaseModel):
    task_id: str = ""
    task_type: str = ""
    # Compatibility mirror until downstream consumers migrate to GraphState.target_spec.
    target_column: str = ""
    target_columns: list[str] = Field(default_factory=list)
    raw_task: str = ""
    dataset_root: str = ""
    target_hint: str = ""
    metric_hint: str = ""
    dataset_paths: DatasetPaths = Field(default_factory=DatasetPaths)
    business_context: str = ""
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)


class DataOverview(StrictBaseModel):
    n_rows: int = 0
    n_cols: int = 0
    column_types: dict[str, str] = Field(default_factory=dict)
    missingness_summary: dict[str, Any] = Field(default_factory=dict)


class TargetEvidence(StrictBaseModel):
    source: str = ""
    detail: str = ""
    columns: list[str] = Field(default_factory=list)


class SubmissionSpec(StrictBaseModel):
    id_columns: list[str] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)
    submission_columns: list[str] = Field(default_factory=list)
    sample_submission_path: str = ""


class EvaluationSpec(StrictBaseModel):
    metric_hint: str = ""
    primary_metric: str = ""
    candidate_metrics: list[str] = Field(default_factory=list)


class TargetSpec(StrictBaseModel):
    status: Literal["resolved", "unresolved"] = "unresolved"
    target_columns: list[str] = Field(default_factory=list)
    # Authoritative: "single_target" / "multi_label" / "multi_output"
    # (per target_spec_migration_plan.md). Legacy "single" / "multi" kept for
    # Reader and pre-migration tests; consumers normalize on read.
    target_mode: Literal[
        "unknown",
        "single",
        "multi",
        "single_target",
        "multi_label",
        "multi_output",
    ] = "unknown"
    task_family: str = ""
    prediction_kind: str = ""
    user_target_hint: str = ""
    user_task_family_hint: str = ""
    submission: SubmissionSpec = Field(default_factory=SubmissionSpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    evidence: list[TargetEvidence] = Field(default_factory=list)


class ProblemUnderstanding(StrictBaseModel):
    task_summary: str = ""
    predicted_task_type: str = ""
    # Compatibility mirror until downstream consumers migrate to GraphState.target_spec.
    target_column: str = ""
    target_columns: list[str] = Field(default_factory=list)
    candidate_metrics: list[str] = Field(default_factory=list)
    data_overview: DataOverview = Field(default_factory=DataOverview)
    risks: list[str] = Field(default_factory=list)
    business_goal_summary: str = ""


class ExecutionPlan(StrictBaseModel):
    phases: list[str] = Field(default_factory=list)
    phase_objectives: dict[str, str] = Field(default_factory=dict)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    review_points: list[str] = Field(default_factory=list)
    fallback_rules: list[str] = Field(default_factory=list)


class DataState(StrictBaseModel):
    raw_train_path: str = ""
    raw_test_path: str = ""
    current_train_path: str = ""
    current_test_path: str = ""
    schema_summary: dict[str, Any] = Field(default_factory=dict)
    cleaning_summary: dict[str, Any] = Field(default_factory=dict)
    eda_findings: dict[str, Any] = Field(default_factory=dict)
    indepth_eda_summary: dict[str, Any] = Field(default_factory=dict)
    data_quality_flags: list[str] = Field(default_factory=list)
    # Compatibility mirror until downstream consumers migrate to GraphState.target_spec.
    target_column: str = ""
    target_columns: list[str] = Field(default_factory=list)
    train_shape: tuple[int, int] | None = None
    test_shape: tuple[int, int] | None = None


class FeatureState(StrictBaseModel):
    engineered_train_path: str = ""
    engineered_test_path: str = ""
    selected_features: list[str] = Field(default_factory=list)
    dropped_features: list[str] = Field(default_factory=list)
    feature_transform_summary: dict[str, Any] = Field(default_factory=dict)
    feature_importance_summary: dict[str, Any] = Field(default_factory=dict)
    leakage_checks: list[str] = Field(default_factory=list)


class ModelState(StrictBaseModel):
    candidate_models: list[str] = Field(default_factory=list)
    best_model_name: str = ""
    best_params: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    cv_results: dict[str, Any] = Field(default_factory=dict)
    prediction_path: str = ""
    model_explanations: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)


class ReviewState(StrictBaseModel):

    
    latest_review_status: Literal["approved", "retry", "fatal"] | None = None

    review_history: list[dict[str, Any]] = Field(default_factory=list)
    retry_count_by_phase: dict[str, int] = Field(default_factory=dict)
    fatal_errors: list[Any] = Field(default_factory=list)
    last_review_comment: str = ""
    pending_review_phase: str = ""


class ReportState(StrictBaseModel):
    technical_report_path: str = ""
    business_report_path: str = ""
    executive_summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class RuntimeState(StrictBaseModel):
    thread_id: str = ""
    run_id: str = ""
    artifact_root: str = ""
    checkpoint_namespace: str = "default"
    token_usage: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = 0
    phase_runtime_sec: dict[str, float] = Field(default_factory=dict)
    total_runtime_sec: float = 0.0
    cost_estimate: float | None = None
    workspace_root: str = ""
    interrupted: bool = False
    last_reviewed_phase: str = ""


class GraphState(StrictBaseModel):
    task_config: TaskConfig = Field(default_factory=TaskConfig)
    problem_understanding: ProblemUnderstanding = Field(default_factory=ProblemUnderstanding)
    target_spec: TargetSpec = Field(default_factory=TargetSpec)
    execution_plan: ExecutionPlan = Field(default_factory=ExecutionPlan)
    current_phase: str = ""
    phase_history: Annotated[list[str], operator.add] = Field(default_factory=list)
    data_state: DataState = Field(default_factory=DataState)
    feature_state: FeatureState = Field(default_factory=FeatureState)
    model_state: ModelState = Field(default_factory=ModelState)
    review_state: ReviewState = Field(default_factory=ReviewState)
    report_state: ReportState = Field(default_factory=ReportState)
    artifacts: dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_ARTIFACT_PATHS))
    runtime_state: RuntimeState = Field(default_factory=RuntimeState)
    messages: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)

    def to_workflow_input(self) -> dict[str, Any]:
        return self.model_dump(mode="python")
