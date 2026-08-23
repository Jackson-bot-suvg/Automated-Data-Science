## Why

The main LangGraph workflow routes through `planner`, but `app/agents/planner.py` is still a placeholder, so approved background review or reviewer retry cannot select the next execution phase. LLM access is also scattered across planner/reviewer-specific code paths; adding a small shared LLM service now lets the PlannerAgent use a standard adapter without forcing other agents to migrate yet.

## What Changes

- Implement a main-chain `PlannerAgent` in `app/agents/planner.py` behind the existing graph node wrapper and `run_planner(state)` entry point.
- Have the planner read `GraphState` evidence and choose the next phase in the diagram order: `preliminary_eda`, `data_cleaning`, `feature_engineering`, `model_building`.
- Update `execution_plan` with phase objectives, success criteria, review points, and fallback rules suitable for the current task.
- Add a reusable `app/services/llm_service.py` adapter for structured or text LLM calls.
- Keep existing agent-specific LLM call sites unchanged for this change, except for the new PlannerAgent.
- Add focused tests for planner phase selection, retry behavior, fallback behavior, and LLM service fallback.

## Capabilities

### New Capabilities
- `main-chain-planner-agent`: Planner node capability that turns shared `GraphState` and reviewer feedback into the next execution phase and a serializable `execution_plan` update.
- `standard-llm-service`: Shared LLM adapter capability that centralizes model configuration, prompt invocation, JSON parsing, and deterministic fallback behavior for planner use.

### Modified Capabilities

## Impact

- Affected code: `app/agents/planner.py`, `app/prompts/planner_prompt.py`, new `app/services/llm_service.py`, and focused tests under `tests/unit/`.
- Existing graph node structure is preserved: `app/graph/nodes/planner_node.py` continues to call `run_planner(state)`.
- No new top-level `GraphState` keys are introduced.
- Existing `BasePlanner` for data-cleaning and feature-engineering tool-call planning is not replaced.
