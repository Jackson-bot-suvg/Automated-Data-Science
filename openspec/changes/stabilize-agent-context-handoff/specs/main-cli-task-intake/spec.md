## ADDED Requirements

### Requirement: CLI intake SHALL preserve stable logical thread identity and separate run identity
The system SHALL preserve a stable logical workflow identity for resume and observability while allocating a unique run identity for per-run artifacts.

#### Scenario: Repeated runs share the same logical thread
- **WHEN** the caller provides the same logical workflow identity for repeated runs of the same task
- **THEN** `runtime_state.thread_id` MUST remain stable across those runs
- **THEN** the workflow invocation MUST continue to pass that stable `thread_id` as the configurable thread identity for graph persistence and tracing

#### Scenario: Each run receives a unique run identity
- **WHEN** the system builds a new initial workflow state
- **THEN** it MUST allocate a unique `run_id` distinct from the logical `thread_id`
- **THEN** the per-run artifact root MUST be derived from `run_id`
- **THEN** the task-level artifact paths for that run MUST resolve inside the run-specific artifact root

### Requirement: Run-specific artifact isolation SHALL NOT rewrite logical workflow identity
Per-run artifact isolation MUST be implemented without rewriting the logical workflow identity used for checkpointing and observability grouping.

#### Scenario: Artifact isolation is enabled
- **WHEN** the workflow enables per-run artifact isolation
- **THEN** the system MUST NOT randomize or replace the logical `thread_id` solely to create unique artifact paths
- **THEN** any run-specific uniqueness required for artifacts MUST come from `run_id` or an equivalent run-scoped field
