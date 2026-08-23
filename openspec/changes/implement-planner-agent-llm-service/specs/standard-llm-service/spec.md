## ADDED Requirements

### Requirement: Shared LLM service centralizes chat model creation
The system SHALL provide a reusable service-layer adapter for LLM calls under `app/services/` so planner code does not instantiate provider-specific chat models directly.

#### Scenario: Planner obtains an LLM response through service layer
- **WHEN** PlannerAgent needs LLM assistance
- **THEN** it MUST call a function or object in `app/services/llm_service.py`
- **THEN** PlannerAgent MUST NOT instantiate `ChatOpenAI` directly

### Requirement: LLM service supports structured JSON responses
The system SHALL provide a service method that accepts system prompt, user prompt, and a deterministic fallback payload, then returns a dictionary parsed from LLM JSON output or the fallback payload.

#### Scenario: Valid JSON response is parsed
- **WHEN** the LLM returns a JSON object containing planner fields
- **THEN** the service MUST return that object as a Python dictionary

#### Scenario: Invalid or unavailable response falls back
- **WHEN** no API key is configured, the service is disabled, the model call fails, or the response is not valid JSON
- **THEN** the service MUST return the caller-provided fallback dictionary

### Requirement: LLM configuration is environment-driven
The system SHALL read model name, API key, base URL, temperature, and max token settings from environment variables with provider-compatible defaults matching the existing project direction.

#### Scenario: No explicit planner environment is provided
- **WHEN** planner-specific LLM environment variables are absent
- **THEN** the service MUST fall back to existing shared variables such as `DASHSCOPE_API_KEY`, `ZHIPUAI_API_KEY`, or `OPENAI_API_KEY`
- **THEN** it MUST preserve a deterministic fallback path if none are available

### Requirement: Existing LLM call sites are not migrated in this change
The system SHALL keep existing `BasePlanner` and reviewer LLM call paths untouched unless a small compatibility import is needed.

#### Scenario: Data cleaning planner remains unchanged
- **WHEN** data cleaning or feature engineering uses `BasePlanner`
- **THEN** that code path MUST continue to work without requiring migration to the new shared LLM service
