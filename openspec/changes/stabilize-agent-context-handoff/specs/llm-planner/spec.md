## ADDED Requirements

### Requirement: Planner outputs SHALL drive executed phase plans
The system SHALL record, for each execution phase, whether the phase used the planner-authored phase plan directly or executed a deterministic fallback plan because the planner bundle was missing or invalid.

#### Scenario: Phase uses the planner-generated plan directly
- **WHEN** a phase agent executes the planner-generated plan without alteration
- **THEN** the phase MUST persist a structured executed-plan record
- **THEN** the record MUST identify the planner plan name, the executed steps, and that the plan source was the planner output

#### Scenario: Phase falls back from the planner plan
- **WHEN** a phase agent cannot use the planner-generated plan because the phase plan is missing or invalid
- **THEN** the phase MUST persist a structured executed-plan record
- **THEN** the record MUST identify that the executed plan came from a deterministic fallback
- **THEN** the record MUST capture the reason the advisory planner plan was not used directly

### Requirement: Reviewer and downstream consumers SHALL validate executed plans, not only advisory plans
The workflow SHALL distinguish between advisory planning state and the plan actually executed by the phase agent.

#### Scenario: Reviewer validates a phase with a fallback plan
- **WHEN** a phase completes using a fallback or regenerated plan
- **THEN** the reviewer MUST be able to inspect the executed-plan record for that phase
- **THEN** the review result MUST reflect the executed plan context rather than assuming the advisory planner plan was followed

#### Scenario: Summaries reference phase execution history
- **WHEN** technical or business reporting summarizes phase behavior
- **THEN** the reporting layer MUST be able to determine which plan source each phase executed
- **THEN** the reporting layer MUST avoid inferring phase execution solely from `execution_plan.success_criteria[*].agent_plan`
