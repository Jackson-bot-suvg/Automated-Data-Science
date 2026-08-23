## ADDED Requirements

### Requirement: Background Reviewer validates Reader output before planning
The system SHALL expose Background Reviewer through the existing `bg_reviewer_node(state)` and `run_bg_reviewer(state)` entry points, with `app/agents/bg_reviewer.py` owning a BackgroundReviewerAgent layer that validates Reader output and writes review state for graph routing.

#### Scenario: Valid Reader output is approved
- **WHEN** Reader output has readable train data, a target column, and non-empty problem understanding
- **THEN** Background Reviewer MUST set `review_state.latest_review_status` to `approved`
- **THEN** it MUST append a review record for phase `reader`
- **THEN** it MUST set `runtime_state.last_reviewed_phase` to `reader`

### Requirement: Background Reviewer distinguishes retryable and fatal failures
The system SHALL route missing or incomplete Reader context to retry, while treating unreadable dataset paths as fatal.

#### Scenario: Missing target is retryable
- **WHEN** the training dataset is readable but no target column can be identified
- **THEN** Background Reviewer MUST set `review_state.latest_review_status` to `retry` until the retry cap is reached
- **THEN** it MUST set `review_state.pending_review_phase` to `reader`

#### Scenario: Missing train file is fatal
- **WHEN** the training dataset path is empty, missing, or unreadable
- **THEN** Background Reviewer MUST set `review_state.latest_review_status` to `fatal`
- **THEN** it MUST add a fatal error and set `runtime_state.interrupted` to true

### Requirement: Background Reviewer comments use deterministic status
The system SHALL keep deterministic checks authoritative and use LLM only to phrase comments.

#### Scenario: LLM returns a conflicting comment
- **WHEN** optional LLM assistance is available for a background review comment
- **THEN** Background Reviewer MUST NOT let the LLM change the deterministic status
- **THEN** it MUST fall back to deterministic comment text if the LLM response is absent or invalid
