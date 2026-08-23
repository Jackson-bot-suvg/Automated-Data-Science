## 1. Contract Authority

- [x] 1.1 Define and document authoritative state ownership per workflow phase in schemas and contracts
- [x] 1.2 Constrain legacy target mirrors to compatibility-only fallback reads in remaining workflow modules
- [x] 1.3 Update downstream consumers to prefer authoritative structured fields before report-text inference

## 2. Run Identity And Artifact Isolation

- [x] 2.1 Standardize `thread_id` as stable logical identity and `run_id` as unique execution identity across intake, CLI, and observability
- [x] 2.2 Ensure all run-scoped artifact roots and artifact keys derive from `run_id`
- [x] 2.3 Add or finalize structured `run_manifest` coverage for the major workflow checkpoints

## 3. Planner And Phase Execution Traceability

- [x] 3.1 Define a structured executed-plan artifact format that records plan source, plan name, steps, and fallback reason
- [x] 3.2 Update execution phases to persist executed-plan records whether they use planner-owned phase plans or deterministic fallbacks
- [x] 3.3 Update reviewer logic to validate against executed-plan context rather than assuming the advisory planner plan was followed

## 4. Review And Reporting Alignment

- [x] 4.1 Persist structured review artifacts or equivalent per-phase review records for debugging and handoff
- [x] 4.2 Update technical summarization to consume structured artifacts for metrics, limitations, and phase provenance
- [x] 4.3 Update business translation to consume structured metrics and driver artifacts before markdown-derived context

## 5. Verification

- [x] 5.1 Add or update unit tests for stable thread identity, unique run identity, executed-plan persistence, and structured review artifacts
- [x] 5.2 Re-run representative workflow tasks to verify context handoff, artifact traceability, and downstream reporting behavior
- [x] 5.3 Record the reduced verification scope when representative-task-only validation is explicitly approved
