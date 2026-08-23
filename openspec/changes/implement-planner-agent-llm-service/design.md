## Context

The graph already has a `planner` node between background review and execution phases, and reviewer retry routes back to planner. However, `app/agents/planner.py` still raises `NotImplementedError`. The current repository also contains multiple LLM call patterns: `app/planners/base.py` directly creates a `ChatOpenAI` structured model for tool-call planning, while reviewer comment generation uses `app/services/reviewer_service.py`.

The requested implementation must follow the project layering pattern:

```text
graph node -> agent.run(state) -> prompt template -> service/tool calls -> state update
```

## Goals / Non-Goals

**Goals:**
- Implement a runnable main-chain PlannerAgent behind existing graph node wrappers.
- Select the next execution phase using current `GraphState`, phase history, and reviewer feedback.
- Standardize a reusable LLM adapter in `app/services/llm_service.py` and use it from PlannerAgent.
- Keep the planner robust when no LLM credentials are configured.
- Add focused unit tests for planner behavior and service fallback.

**Non-Goals:**
- Do not replace `app/planners/base.py`; it remains responsible for data-cleaning and feature-engineering tool-call plans.
- Do not migrate reviewer, data-cleaning, or feature-engineering LLM call sites in this change.
- Do not change graph edges unless a wiring bug is found in the existing planner node wrapper.
- Do not add new top-level `GraphState` fields.

## Decisions

### Decision 1: Keep `PlannerAgent` in `planner.py`

`app/agents/planner.py` will contain both the public entry point `run_planner(state)` and the `PlannerAgent` implementation. This keeps the agent contract explicit without introducing a second planner module for the same capability.

Alternative considered: create `app/agents/planner_agent.py` and have `planner.py` delegate to it. Rejected because this project already treats `app/agents/<role>.py` as the agent file, and keeping two planner files adds an unnecessary dependency hop.

### Decision 2: Planner LLM output is advisory, deterministic fallback is authoritative

The LLM may propose `next_phase`, objectives, criteria, and fallback rules. The agent validates `next_phase` against the supported phase set and falls back to deterministic phase order when the response is absent or invalid.

Alternative considered: require LLM to choose every phase. Rejected because main graph routing must work in local test environments without API credentials.

### Decision 3: Shared LLM service returns parsed JSON with caller fallback

The new service will expose a small `complete_json(...)`-style function. It will read provider settings from environment variables, call `ChatOpenAI` when possible, parse fenced or plain JSON, and return the caller-provided fallback payload on any failure.

Alternative considered: introduce a broad LLM client abstraction and migrate all agents immediately. Rejected because this task only needs planner integration, and broad migration would create unnecessary risk.

### Decision 4: Planner phase order is explicit and local

The PlannerAgent will use the execution phase sequence:

```text
preliminary_eda -> data_cleaning -> feature_engineering -> model_building
```

It intentionally skips `indepth_eda` for MVP because the in-depth EDA node is still a placeholder and would block runnable main-chain progress.

## Risks / Trade-offs

- **Risk:** The graph contains an `indepth_eda` route but the planner skips it for MVP. **Mitigation:** Document this as an MVP decision and keep the router capable of accepting `indepth_eda` later.
- **Risk:** LLM output can request an invalid phase. **Mitigation:** Validate against the supported phase list and use deterministic fallback.
- **Risk:** Adding a shared LLM service could tempt large migration. **Mitigation:** Only PlannerAgent uses it in this change; existing call sites remain untouched.
- **Risk:** `phase_history` reducer can duplicate phases if nodes return old history plus new phase. **Mitigation:** Planner returns only `[next_phase]` for the reducer.

## Migration Plan

1. Add `app/services/llm_service.py` with JSON completion and fallback behavior.
2. Implement `PlannerAgent` and `run_planner(state)` in `app/agents/planner.py`.
3. Keep `app/graph/nodes/planner_node.py` and `app/graph/workflow.py` unchanged unless tests reveal wiring issues.
4. Add unit tests and run planner-focused tests.

## Open Questions

- Whether `indepth_eda` should become part of the default planner phase order after its node is implemented.
- Whether a later change should migrate `reviewer_service.py` and `BasePlanner` to the shared LLM service.
