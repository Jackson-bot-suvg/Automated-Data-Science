## 1. Reader Agent

- [x] 1.1 Implement `ReaderAgent` in `app/agents/reader.py` with bounded CSV metadata inspection, deterministic fallback, optional `READER_LLM` JSON refinement, and `run_reader(state)` entry point.
- [x] 1.2 Update `app/prompts/reader_prompt.py` with a structured Reader prompt builder and JSON output contract.
- [x] 1.3 Add unit tests for Reader state updates, deterministic fallback, inferred task metadata, and missing-path risk handling.

## 2. Background Reviewer Agent

- [x] 2.1 Implement `BackgroundReviewerAgent` in `app/agents/bg_reviewer.py` with deterministic Reader-output checks, retry/fatal handling, review history append, and optional LLM comment phrasing.
- [x] 2.2 Update `app/prompts/reviewer_prompt.py` with background reviewer comment prompt support.
- [x] 2.3 Add unit tests for approved, retry, and fatal background review outcomes.

## 3. Technical Summarizer Agent

- [x] 3.1 Implement `TechnicalSummarizerAgent` in `app/agents/summarizer.py` with final-only report generation, deterministic report fallback, optional `SUMMARIZER_LLM` JSON refinement, and `run_technical_summary(state)` entry point.
- [x] 3.2 Update `app/prompts/summarizer_prompt.py` with a structured technical summarizer prompt builder and JSON output contract.
- [x] 3.3 Update `route_after_summarizer` only as needed to support `current_phase == "technical_summary"` routing to business translation.
- [x] 3.4 Add unit tests for non-final no-op behavior, final technical report creation, report state/artifact updates, and summarizer route compatibility.

## 4. Verification

- [x] 4.1 Run focused tests for Reader, Background Reviewer, Technical Summarizer, planner, reviewer, and business translator.
- [x] 4.2 Run full unit tests and OpenSpec status/apply checks, then update this task list as complete.
