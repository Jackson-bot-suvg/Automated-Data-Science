# I/O Contract Quick Reference

This document is a fast implementation index for the LangGraph engineering contract.
The source of truth is:

- `docs/engineering-specification.md`
- `app/schemas/state_schema.py`
- `app/schemas/io_contracts.py`

If there is any conflict, follow the engineering specification and schema modules.

## 1. Graph Runtime Rules

- Build the workflow with `StateGraph(GraphState)`.
- Start from `START -> reader`.
- End at `business_translation -> END`.
- Use conditional routing for review gates and retries.
- Compile with a checkpointer.
- Invoke with `configurable.thread_id`.

## 2. Shared State Keys

```python
{
    "task_config": {},
    "problem_understanding": {},
    "execution_plan": {},
    "current_phase": "",
    "phase_history": [],
    "data_state": {},
    "feature_state": {},
    "model_state": {},
    "review_state": {},
    "report_state": {},
    "artifacts": {},
    "runtime_state": {},
    "messages": []
}
```

## 3. Reducer-Sensitive Fields

| Key | Rule |
| --- | --- |
| `phase_history` | append-only (handled by Annotated reducer) |
| `review_state.review_history` | append-only (requires MANUAL merging by node) |
| `artifacts.figures` | append-only (requires MANUAL merging by node) |
| `messages` | append-only (handled by Annotated reducer) |

> [!WARNING]
> Top level fields like `phase_history` and `messages` use `Annotated[list, operator.add]` so returning a new element array will safely append.
> However, `review_history` and `figures` are nested. To keep the schema simple (K.I.S.S), no deep-merge reducers are defined. Node developers MUST manually retrieve the existing lists, append, and formulate the complete list when returning these nested state updates.

Do not introduce parallel branches that write to these keys until reducer-safe merging is defined.

## 4. Node Contract Snapshot

| Node | Entry | Required Read Set | Required Write Set |
| --- | --- | --- | --- |
| Reader | `run_reader(state: GraphState) -> dict` | `task_config` | `problem_understanding`, `current_phase`, `phase_history` |
| Planner | `run_planner(state: GraphState) -> dict` | `task_config`, `problem_understanding` | `execution_plan`, `current_phase`, `phase_history` |
| Preliminary EDA | `run_preliminary_eda(state: GraphState) -> dict` | `data_state.current_train_path`, `problem_understanding.target_column` | `data_state.schema_summary`, `data_state.eda_findings`, `artifacts.preliminary_eda_report`, `artifacts.figures` |
| Data Cleaning | `run_data_cleaning(state: GraphState) -> dict` | `data_state.current_train_path`, `data_state.current_test_path`, `data_state.eda_findings`, `problem_understanding.target_column` | `data_state.current_train_path`, `data_state.current_test_path`, `data_state.cleaning_summary`, `artifacts.cleaned_train_data`, `artifacts.cleaned_test_data`, `artifacts.cleaning_report` |
| Feature Engineering | `run_feature_engineering(state: GraphState) -> dict` | `data_state.current_train_path`, `data_state.current_test_path`, `data_state.eda_findings` | `feature_state.engineered_train_path`, `feature_state.engineered_test_path`, `feature_state.selected_features`, `feature_state.feature_transform_summary`, `artifacts.feature_summary` |
| Modeling | `run_model_building(state: GraphState) -> dict` | `feature_state.engineered_train_path`, `feature_state.engineered_test_path`, `problem_understanding.predicted_task_type`, `problem_understanding.candidate_metrics` | `model_state.candidate_models`, `model_state.best_model_name`, `model_state.validation_metrics`, `model_state.prediction_path`, `artifacts.model_results`, `artifacts.best_model`, `artifacts.prediction_file` |
| Reviewer | `run_reviewer(state: GraphState) -> dict` | `current_phase`, `execution_plan.success_criteria`, `artifacts` | `review_state.latest_review_status`, `review_state.last_review_comment`, `review_state.review_history`, `review_state.pending_review_phase`, `runtime_state.last_reviewed_phase`, `runtime_state.interrupted` |
| Technical Summary | `run_technical_summary(state: GraphState) -> dict` | `data_state`, `feature_state`, `model_state`, `review_state` | `report_state.technical_report_path`, `artifacts.technical_report` |
| Business Translation | `run_business_translation(state: GraphState) -> dict` | `report_state.technical_report_path`, `model_state.validation_metrics`, `feature_state.feature_importance_summary`, `task_config.business_context` | `report_state.business_report_path`, `report_state.executive_summary`, `report_state.recommended_actions`, `artifacts.business_report` |

## 5. Artifact Keys

```text
preliminary_eda_report
cleaned_train_data
cleaned_test_data
cleaning_report
in_depth_eda_report
feature_summary
model_results
best_model
prediction_file
technical_report
business_report
figures
```

## 6. Runtime Fields Required for LangGraph

```python
{
    "thread_id": "",
    "checkpoint_namespace": "default",
    "workspace_root": "",
    "interrupted": False,
    "last_reviewed_phase": "",
}
```

## 7. Reviewer Decisions

Allowed reviewer status values:

- `approved`
- `retry`
- `fatal`

Reviewer gates apply to:

- `preliminary_eda`
- `data_cleaning`
- `indepth_eda`
- `feature_engineering`
- `model_building`
- `technical_summary`
- `business_translation`

## 8. Observability

- LangSmith tracing is a side-channel capability, not part of `GraphState`.
- Node contracts do not change when tracing is enabled.
- Tracing metadata must be derived from existing runtime inputs, state, and environment configuration.
- Tracing must not introduce new top-level state keys.
