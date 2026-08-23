Multi-Agent Automated Data Science System

LangGraph Engineering Specification (v2.0)

1. Purpose

This document defines the minimum engineering contract for a LangGraph-based multi-agent data science system.
It is the source of truth for:

- shared state boundaries
- node input and output contracts
- reducer rules for append-only fields
- routing, review gates, and persistence
- artifact naming
- repository structure
- delivery and test expectations

Code that does not comply with the state schema, node contracts, reducer rules, or artifact naming rules is not considered complete.

2. Core Principles

2.1 Shared State First

All node communication must go through `GraphState`.
Direct parameter passing between business nodes is not allowed.
Node outputs must be merged back into the shared state.

2.2 LangGraph Native Orchestration

The workflow must be modeled as a `StateGraph(GraphState)`.
Execution entry and exit must be represented through `START` and `END`.
Conditional transitions must use `add_conditional_edges(...)` or `Command(...)`.

2.3 Explicit Persistence

The compiled graph must use a checkpointer.
Each invocation must provide `configurable.thread_id`.
Reviewer approval and any human-in-the-loop pause must resume through durable execution rather than ad hoc in-memory flags.

2.4 Reducer Discipline

Any state field that can be appended to by more than one node or by repeated execution attempts must have an explicit reducer rule.
If no reducer is defined, the field must be treated as overwrite-only and must not be updated concurrently.

2.5 Evidence over Claims

No node may declare success without evidence.
Evidence must be represented in at least two of the following:

- updated state
- generated artifact
- structured log

3. Graph Runtime Contract

The graph must follow this runtime contract:

- Builder: `StateGraph(GraphState)`
- Entry edge: `START -> reader`
- Terminal edge: `business_translation -> END`
- Review gates: `data_cleaning`, `model_building`, `technical_summary`, and `business_translation`
- Routing: `add_conditional_edges(...)` or `Command(...)`
- Persistence: compile with a checkpointer
- Invocation: pass `configurable.thread_id`
- Resume: resume by thread identifier after reviewer or human approval

4. Unified GraphState Structure

`GraphState` must contain only the following top-level keys:

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

Top-level keys are closed by default.
Unauthorized top-level additions are forbidden.

4.1 task_config

```python
{
    "task_id": "",
    "task_type": "",
    "target_column": "",
    "metric_hint": "",
    "dataset_paths": {
        "train": "",
        "test": "",
        "submission": "",
    },
    "business_context": "",
    "constraints": {
        "max_runtime_min": None,
        "max_tokens": None,
        "interpretability_required": True,
        "business_translation_required": True,
    },
}
```

4.2 problem_understanding

```python
{
    "task_summary": "",
    "predicted_task_type": "",
    "target_column": "",
    "candidate_metrics": [],
    "data_overview": {
        "n_rows": 0,
        "n_cols": 0,
        "column_types": {},
        "missingness_summary": {},
    },
    "risks": [],
    "business_goal_summary": "",
}
```

4.3 execution_plan

```python
{
    "phases": [],
    "phase_objectives": {},
    "success_criteria": {},
    "review_points": [],
    "fallback_rules": [],
}
```

4.4 data_state

```python
{
    "raw_train_path": "",
    "raw_test_path": "",
    "current_train_path": "",
    "current_test_path": "",
    "schema_summary": {},
    "cleaning_summary": {},
    "eda_findings": [],
    "data_quality_flags": [],
    "target_column": "",
    "train_shape": None,
    "test_shape": None,
}
```

4.5 feature_state

```python
{
    "engineered_train_path": "",
    "engineered_test_path": "",
    "selected_features": [],
    "dropped_features": [],
    "feature_transform_summary": {},
    "feature_importance_summary": {},
    "leakage_checks": [],
}
```

4.6 model_state

```python
{
    "candidate_models": [],
    "best_model_name": "",
    "best_params": {},
    "validation_metrics": {},
    "cv_results": {},
    "prediction_path": "",
    "model_explanations": {},
    "baseline_metrics": {},
}
```

4.7 review_state

```python
{
    "latest_review_status": "",
    "review_history": [],
    "retry_count_by_phase": {},
    "fatal_errors": [],
    "last_review_comment": "",
    "pending_review_phase": "",
}
```

4.8 report_state

```python
{
    "technical_report_path": "",
    "business_report_path": "",
    "executive_summary": "",
    "limitations": [],
    "recommended_actions": [],
}
```

4.9 artifacts

```python
{
    "preliminary_eda_report": "",
    "cleaned_train_data": "",
    "cleaned_test_data": "",
    "cleaning_report": "",
    "in_depth_eda_report": "",
    "feature_summary": "",
    "model_results": "",
    "best_model": "",
    "prediction_file": "",
    "technical_report": "",
    "business_report": "",
    "figures": [],
}
```

4.10 runtime_state

```python
{
    "thread_id": "",
    "checkpoint_namespace": "default",
    "token_usage": {},
    "llm_calls": 0,
    "phase_runtime_sec": {},
    "total_runtime_sec": 0.0,
    "cost_estimate": None,
    "workspace_root": "",
    "interrupted": False,
    "last_reviewed_phase": "",
}
```

4.11 messages

`messages` is reserved for serialized LLM or agent conversation records.
It is append-only.
The project may later switch to LangChain message objects, but any stored representation must remain serializable and safe for checkpointing.

5. Reducer Rules

The following keys are append-only and must not be treated as blind overwrite fields:

| Key | Rule | Reducer Implementation |
| --- | --- | --- |
| `phase_history` | append-only | `Annotated[list, operator.add]` (automatic) |
| `review_state.review_history` | append-only | Manual array concatenation by the Node |
| `artifacts.figures` | append-only | Manual array concatenation by the Node |
| `messages` | append-only messages | `Annotated[list, operator.add]` (automatic) |

因为 LangGraph 默认 reducer 仅作用于顶层(Top-level)键，诸如 `review_history` 等嵌套在字典中的列表尚未实现深度合并。节点在更新嵌套字段时，必须手动加载历史并追加，严禁直接返回盲覆盖结果。

If parallel branches are introduced, reducer-compatible merging must be implemented before those branches are enabled.

6. Node Contract

All business nodes must expose the following interface shape:

- `run_reader(state: GraphState) -> dict`
- `run_planner(state: GraphState) -> dict`
- `run_preliminary_eda(state: GraphState) -> dict`
- `run_data_cleaning(state: GraphState) -> dict`
- `run_feature_engineering(state: GraphState) -> dict`
- `run_model_building(state: GraphState) -> dict`
- `run_reviewer(state: GraphState) -> dict`
- `run_technical_summary(state: GraphState) -> dict`
- `run_business_translation(state: GraphState) -> dict`

Nodes must return partial state updates that can be merged into `GraphState`.
Routing decisions belong to graph routing functions, not to ad hoc second function arguments.

7. Routing and Review Gate Rules

The baseline phase order is:

`reader -> planner -> preliminary_eda -> data_cleaning -> feature_engineering -> model_building -> technical_summary -> business_translation`

The following phases are review gated:

- `data_cleaning`
- `model_building`
- `technical_summary`
- `business_translation`

The reviewer decides one of:

- `approved`
- `revise`
- `failed`

Required reviewer side effects:

- update `review_state.latest_review_status`
- append to `review_state.review_history`
- update `review_state.pending_review_phase`
- update `runtime_state.last_reviewed_phase`
- set `runtime_state.interrupted` when execution must pause

8. Error Contract

All node failures must use the standard error envelope:

```python
{
    "status": "error",
    "phase": "data_cleaning",
    "error_type": "runtime_error",
    "message": "Column not found: target",
    "details": {
        "traceback": "...",
        "failed_artifact": "artifacts/data/cleaned_train.csv",
    },
}
```

Supported error types:

- `runtime_error`
- `logic_error`
- `schema_error`
- `data_leakage_risk`
- `artifact_missing`
- `validation_failure`

9. Artifact Naming

The following artifact paths are reserved and must not be renamed:

```text
artifacts/data/raw_train.csv
artifacts/data/raw_test.csv
artifacts/data/cleaned_train.csv
artifacts/data/cleaned_test.csv
artifacts/data/cleaning_report.md
artifacts/eda/preliminary_eda.md
artifacts/eda/in_depth_eda.md
artifacts/features/engineered_train.csv
artifacts/features/engineered_test.csv
artifacts/features/feature_summary.json
artifacts/models/model_results.json
artifacts/models/best_model.pkl
artifacts/predictions/submission.csv
artifacts/reports/technical_report.md
artifacts/reports/business_report.md
artifacts/figures/<phase>_<name>.png
```

10. Repository Structure

The repository must use the following structure:

```text
project/
├─ app/
│  ├─ graph/
│  │  ├─ __init__.py
│  │  ├─ state.py
│  │  ├─ workflow.py
│  │  ├─ nodes.py
│  │  └─ routes.py
│  ├─ agents/
│  │  ├─ __init__.py
│  │  ├─ reader.py
│  │  ├─ planner.py
│  │  ├─ developer.py
│  │  ├─ reviewer.py
│  │  ├─ summarizer.py
│  │  └─ business_translator.py
│  ├─ llm/
│  │  ├─ __init__.py
│  │  ├─ config.py
│  │  ├─ router.py
│  │  ├─ schemas.py
│  │  ├─ service.py
│  │  └─ providers/
│  │     ├─ __init__.py
│  │     ├─ base.py
│  │     ├─ openai_compat.py
│  │     └─ anthropic.py
│  ├─ tools/
│  │  ├─ __init__.py
│  │  ├─ file_tools.py
│  │  ├─ exec_tools.py
│  │  ├─ metric_tools.py
│  │  ├─ artifact_tools.py
│  │  └─ plot_tools.py
│  ├─ schemas/
│  │  ├─ __init__.py
│  │  ├─ state_schema.py
│  │  ├─ io_contracts.py
│  │  ├─ artifact_schema.py
│  │  └─ error_schema.py
│  └─ prompts/
│     ├─ __init__.py
│     ├─ reader_prompt.py
│     ├─ planner_prompt.py
│     ├─ developer_prompt.py
│     ├─ reviewer_prompt.py
│     └─ business_prompt.py
├─ artifacts/
├─ benchmarks/
├─ docs/
│  ├─ architecture.md
│  ├─ engineering-specification.md
│  ├─ skills/
│  └─ io_contracts.md
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ fixtures/
└─ main.py
```

11. Delivery Checklist

Before any node or module is considered ready, all of the following must be true:

- the file exists in the correct package
- the public entry function is declared
- the state read and write boundaries are documented
- reserved artifact names are used
- at least one unit test or contract smoke test exists
- the code does not mutate fields owned by unrelated modules

12. Minimum Test Baseline

The repository must keep:

- `tests/integration/test_state_smoke.py`
- `tests/integration/test_minimal_workflow.py`

The minimal integration baseline must verify:

- the baseline phase sequence is preserved
- required artifacts are generated with reserved names
- `GraphState` exposes the mandatory top-level keys
- reviewer outputs can be recorded without schema mismatch
- runtime fields required for LangGraph persistence are present

13. Hard Constraints

- Do not add top-level state keys without architecture approval.
- Do not rename reserved artifact keys or paths.
- Do not add non-serializable objects to checkpointed state.
- Do not bypass reviewer gates through direct state mutation.
- Do not implement parallel fan-out until reducer semantics are explicitly added.

14. Language and Encoding

All code and documentation files must use UTF-8 without BOM.
Code identifiers, file names, and contract definitions must remain in English.

15. Observability Contract

- Tracing and monitoring integrations are infrastructure concerns and must not be implemented directly inside business node logic.
- LangSmith metadata and tags may be derived from existing state, runtime config, and environment variables, but tracing must not write back into `GraphState`.
- Tracing failures must degrade safely and must not become business-logic branch conditions.
- Enabling or disabling tracing must not change node signatures, artifact keys, or workflow topology.
