## Why

The main LangGraph workflow now has a runnable planner and reviewer path, but the upstream Reader, Background Reviewer, and downstream Technical Summarizer are still placeholders. Without these three nodes, the graph cannot turn an incoming task and dataset paths into validated problem context, nor can it produce a technical report before business translation.

## What Changes

- Implement `ReaderAgent` in `app/agents/reader.py` behind the existing `run_reader(state)` entry point.
- Implement `BackgroundReviewerAgent` in `app/agents/bg_reviewer.py` behind the existing `run_bg_reviewer(state)` entry point.
- Implement `TechnicalSummarizerAgent` in `app/agents/summarizer.py` behind the existing `run_technical_summary(state)` entry point.
- Update reader, background reviewer, and summarizer prompts to assemble bounded context for optional LLM assistance through the shared `app/services/llm_service.py`.
- Keep graph/node wiring stable: existing node wrappers continue to call the agent entry functions.
- Add focused tests covering deterministic fallback, retry/fatal behavior, technical report creation, and graph router compatibility.

## Capabilities

### New Capabilities
- `reader-agent`: Converts `task_config` and dataset paths into structured `problem_understanding` and initial `data_state`.
- `background-reviewer-agent`: Validates Reader output before planner execution and writes review state for approved/retry/fatal routing.
- `technical-summarizer-agent`: Produces the final technical report artifact after model-building approval while preserving current graph routing semantics.

### Modified Capabilities

## Impact

- Affected code: `app/agents/reader.py`, `app/agents/bg_reviewer.py`, `app/agents/summarizer.py`, `app/prompts/reader_prompt.py`, `app/prompts/reviewer_prompt.py`, `app/prompts/summarizer_prompt.py`, and focused unit tests under `tests/unit/`.
- Existing graph node files remain thin wrappers and should not need structural changes.
- The shared LLM service remains optional; these nodes must remain runnable without API credentials.
- No new top-level `GraphState` fields are introduced.
