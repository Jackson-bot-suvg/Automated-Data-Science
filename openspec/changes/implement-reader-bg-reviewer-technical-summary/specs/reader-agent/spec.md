## ADDED Requirements

### Requirement: Reader converts task input into workflow state
The system SHALL expose Reader through the existing `reader_node(state)` and `run_reader(state)` entry points, with `app/agents/reader.py` owning a ReaderAgent layer that reads `GraphState.task_config`, inspects bounded dataset metadata, and returns a serializable partial state update.

#### Scenario: Reader initializes problem and data state
- **WHEN** `run_reader(state)` receives `task_config` with readable train/test dataset paths
- **THEN** it MUST populate `problem_understanding`
- **THEN** it MUST populate raw and current train/test paths in `data_state`
- **THEN** it MUST return `current_phase` equal to `reader`
- **THEN** it MUST append `reader` to `phase_history`

### Requirement: Reader produces deterministic fallback without LLM credentials
The system SHALL produce valid Reader output without requiring any LLM API key.

#### Scenario: LLM is disabled
- **WHEN** `READER_LLM_ENABLED` is disabled or no API key is configured
- **THEN** Reader MUST infer target, task type, candidate metrics, data overview, and risks using deterministic dataset metadata
- **THEN** Reader MUST NOT raise solely because LLM assistance is unavailable

### Requirement: Reader keeps context bounded
The system SHALL avoid loading full datasets into the LLM context.

#### Scenario: Reader builds LLM context
- **WHEN** Reader requests optional LLM assistance
- **THEN** it MUST send bounded schema, shape, missingness, explicit task config, and small sample metadata
- **THEN** it MUST NOT include entire CSV contents in the prompt
