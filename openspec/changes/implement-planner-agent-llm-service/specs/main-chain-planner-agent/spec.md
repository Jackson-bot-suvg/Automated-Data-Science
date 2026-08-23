## ADDED Requirements

### Requirement: Planner runs as a main-chain LangGraph node
The system SHALL expose the planner through the existing `planner_node(state)` and `run_planner(state)` entry points, with `app/agents/planner.py` owning the PlannerAgent layer that reads `GraphState`, uses prompt/service dependencies, and returns a serializable partial state update.

#### Scenario: Planner node delegates to PlannerAgent
- **WHEN** `planner_node(state)` is invoked by the LangGraph workflow
- **THEN** it MUST call `run_planner(state)`
- **THEN** `run_planner(state)` MUST call `PlannerAgent().run(state)` and return that state update

### Requirement: Planner selects the next executable phase
The system SHALL select one of `preliminary_eda`, `data_cleaning`, `feature_engineering`, or `model_building` as `current_phase` based on the current shared state, previous phase history, and reviewer retry feedback.

#### Scenario: Initial planner call chooses preliminary EDA
- **WHEN** the planner runs after background review approval and no execution phase has completed
- **THEN** it MUST return `current_phase` equal to `preliminary_eda`
- **THEN** it MUST append `preliminary_eda` to `phase_history`

#### Scenario: Approved EDA advances to data cleaning
- **WHEN** `preliminary_eda` has completed and the latest review status is `approved`
- **THEN** the planner MUST return `current_phase` equal to `data_cleaning`

#### Scenario: Retry keeps focus on the failed phase
- **WHEN** the latest review status is `retry` and `review_state.pending_review_phase` is one of the supported execution phases
- **THEN** the planner MUST return `current_phase` equal to that pending review phase

### Requirement: Planner writes execution plan context
The system SHALL update `execution_plan` with the planned phase sequence, phase objectives, success criteria, review points, and fallback rules without adding new top-level `GraphState` keys.

#### Scenario: Execution plan is serializable
- **WHEN** `run_planner(state)` returns successfully
- **THEN** the returned `execution_plan` MUST be a dictionary accepted by `ExecutionPlan`
- **THEN** the returned top-level keys MUST be limited to `execution_plan`, `current_phase`, `phase_history`, and optionally `messages`

### Requirement: Planner degrades deterministically when LLM is unavailable
The system SHALL produce a valid next phase and execution plan even when the LLM service has no API key, is disabled, or returns malformed output.

#### Scenario: LLM fallback still routes the graph
- **WHEN** the LLM service cannot produce structured planner output
- **THEN** the planner MUST use deterministic phase-order rules to produce a valid `current_phase`
- **THEN** the workflow router MUST still be able to route after planner
