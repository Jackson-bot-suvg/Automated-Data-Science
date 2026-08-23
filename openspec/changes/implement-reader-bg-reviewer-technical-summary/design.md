## Context

The LangGraph workflow already wires `reader -> background_reviewer -> planner` and `reviewer -> summarizer -> business_translator/planner`, but `app/agents/reader.py`, `app/agents/bg_reviewer.py`, and `app/agents/summarizer.py` are placeholders. The planner and main reviewer are now runnable, and the shared `app/services/llm_service.py` can provide optional JSON LLM assistance with deterministic fallback.

The graph has one important routing nuance: `summarizer` is currently invoked after every approved execution phase, and `route_after_summarizer` decides whether to return to planner or continue to business translation. The technical summarizer implementation therefore must not write a final report before all execution phases are complete.

## Goals / Non-Goals

**Goals:**
- Implement runnable Reader, Background Reviewer, and Technical Summarizer agents behind existing node wrappers.
- Keep each agent aligned with the project layering pattern: `graph node -> run_* entrypoint -> Agent.run(state) -> prompt/service/tool helpers -> partial state update`.
- Use deterministic logic first so the graph runs without API credentials.
- Use the shared LLM service only as optional enhancement for structured problem understanding, review comments, and report text.
- Preserve LangGraph routing semantics and append-only reducer expectations.

**Non-Goals:**
- Do not change the core execution phase agents (`preliminary_eda`, `data_cleaning`, `feature_engineering`, `model_building`).
- Do not replace the main `ReviewerAgent`; background review has a narrower pre-planner contract.
- Do not add new top-level `GraphState` fields.
- Do not require new external dependencies.

## Decisions

### Decision 1: Reader owns deterministic task understanding with optional LLM refinement

`ReaderAgent` will read `task_config.dataset_paths`, load small CSV metadata using pandas, infer target/task type/metrics when missing, and populate `problem_understanding` plus initial `data_state`. It will call `complete_json(..., env_prefix="READER_LLM")` with a bounded context and use deterministic fallback when LLM output is unavailable or invalid.

Alternative considered: make Reader fully LLM-driven. Rejected because the first node must work offline and must not block the graph on credentials.

### Decision 2: Background Reviewer is a narrow pre-planner reviewer

`BackgroundReviewerAgent` will validate Reader output and dataset readability before the planner runs. It writes `review_state.latest_review_status`, appends a `ReviewRecord`, tracks retry attempts for phase `reader`, and marks fatal path/read errors. It may use the shared LLM service to phrase comments, but deterministic checks remain authoritative.

Alternative considered: reuse `ReviewerAgent`. Rejected because `ReviewerAgent` reviews execution phases after artifacts exist; background review needs a different rule set focused on task/context readiness.

### Decision 3: Technical Summarizer is final-only under current workflow wiring

`TechnicalSummarizerAgent` will no-op when called after an approved non-final phase and leave `current_phase` unchanged so `route_after_summarizer` returns `next_phase`. When called after approved `model_building`, it writes `artifacts/reports/technical_report.md`, updates `report_state`, appends `technical_summary` to `phase_history`, and sets `current_phase` to `technical_summary`. The router will treat both `model_building` and `technical_summary` as final-ready.

Alternative considered: change workflow edges so `summarizer` only runs after a separate "all phases done" router. Rejected for this change because node/graph scaffolding already exists and can be made runnable with a small compatible agent/router adjustment.

### Decision 4: LLM service remains JSON-only for this change

All three agents will use `complete_json` for optional LLM assistance. Report/comment text can still be returned inside a JSON field. This keeps one shared service contract and avoids adding a second text-completion API before multiple agents need it.

Alternative considered: add `complete_text`. Rejected because the existing service is sufficient for bounded structured responses and fallback behavior.

## Risks / Trade-offs

- **Risk:** Reader inference can misclassify unusual tasks. **Mitigation:** Prefer explicit `task_config` fields over inference and record uncertainty in `problem_understanding.risks`.
- **Risk:** Background review can retry when user input is actually wrong. **Mitigation:** File missing/unreadable errors become fatal; missing target/context gets retry up to the retry cap.
- **Risk:** Summarizer being called after each approved phase can create duplicate messages. **Mitigation:** Non-final calls return a small skip message and no report artifact.
- **Risk:** LLM output can be malformed or too broad. **Mitigation:** Validate fields and fall back to deterministic payloads.

## Migration Plan

1. Implement Reader, Background Reviewer, and Technical Summarizer agents in the existing `app/agents/*.py` files.
2. Update prompt builders in `app/prompts/reader_prompt.py`, `app/prompts/reviewer_prompt.py`, and `app/prompts/summarizer_prompt.py`.
3. Adjust `route_after_summarizer` to route `technical_summary` to the final business translator path.
4. Add focused unit tests and run targeted plus full unit test suites.
