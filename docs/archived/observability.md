# Observability and LangSmith

This project uses LangSmith as an optional observability layer for LLM and LangGraph execution.

## Scope

Current tracing coverage is intentionally limited to infrastructure entry points:

- graph invocation
- `BasePlanner` LLM calls

Business agents must remain unaware of LangSmith-specific details.

## Project `.env` Configuration

Tracing and LLM provider settings are maintained in the repository root `.env` file.
Because this is a private repository, `.env` is tracked rather than ignored.

Process environment values can still override `.env` values for tests and CI, but normal local development should edit `.env`.

Tracing is opt-in:

- `LANGSMITH_TRACING=true|false`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=26s1-cs33-2-dev`
- `LANGSMITH_WORKSPACE_ID=...`
- `APP_ENV=dev|test|prod`
- `DASHSCOPE_API_KEY=...`
- `ZHIPUAI_API_KEY=...`
- `DASHSCOPE_BASE_URL=...`
- `OPENAI_API_KEY=...`
- `OPENAI_BASE_URL=...`
- `ANTHROPIC_API_KEY=...`
- `ANTHROPIC_BASE_URL=...`

The `app/llm/` service layer reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_BASE_URL` for provider configuration.

If `LANGSMITH_TRACING` is not `true`, tracing helpers become no-ops.
If tracing is requested but LangSmith is not correctly configured, the runtime degrades safely instead of changing business behavior.

## Metadata and Tags

### Workflow traces

Workflow metadata should include:

- `app_name`
- `workflow`
- `thread_id`
- `current_phase`
- `task_type`
- `target_column`
- `dataset`
- `train_path`
- `test_path`

Workflow tags should include:

- `langgraph`
- `planner`
- `dataset:<name>` when the dataset can be inferred
- `env:<app_env>`

### Planner traces

Planner metadata should include:

- `ls_provider`
- `ls_model_name`
- `planner_type`
- `phase`
- `temperature`
- `max_tokens`
- `target_column`
- `dataset_shape`

Planner tags should include:

- `planner`
- `llm`
- `phase:<planner_type>`

## Local Debugging

1. Fill the required LangSmith and provider values in `.env`.
2. Run the project entrypoint or planner flow as usual.
3. Inspect traces in the configured LangSmith project.

Tracing must never be required for normal local development or testing.
