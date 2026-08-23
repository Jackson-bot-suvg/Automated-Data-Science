## 1. Shared LLM Service

- [x] 1.1 Add `app/services/llm_service.py` with environment-driven model configuration and JSON completion fallback.
- [x] 1.2 Export the shared LLM service entry point from `app/services/__init__.py`.
- [x] 1.3 Add unit coverage for JSON parsing and fallback behavior when LLM is disabled or unavailable.

## 2. Planner Agent

- [x] 2.1 Implement `PlannerAgent` in `app/agents/planner.py` with deterministic phase-order fallback and LLM-assisted plan parsing.
- [x] 2.2 Keep `run_planner(state)` in `app/agents/planner.py` as the graph-facing entry point that calls `PlannerAgent().run(state)`.
- [x] 2.3 Update `app/prompts/planner_prompt.py` with a main-chain planner prompt builder that only injects state summaries.
- [x] 2.4 Ensure planner state updates are limited to `execution_plan`, `current_phase`, `phase_history`, and optional `messages`.

## 3. Tests And Verification

- [x] 3.1 Add planner unit tests for initial phase selection, approved-phase advance, reviewer retry behavior, and invalid LLM fallback.
- [x] 3.2 Add a graph/router smoke test confirming planner output can route through `route_after_planner`.
- [x] 3.3 Run focused tests and update this task list as tasks complete.
