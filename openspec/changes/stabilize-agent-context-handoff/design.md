## Context

The current workflow passes a complete `GraphState` through every node, but real ownership is only implicit in code. Several problems follow from that:

- `target_spec` is the intended authority, yet legacy target mirrors still participate in primary behavior and reviewer fallback.
- The phase-level `planner` writes `execution_plan.success_criteria[*].agent_plan`, but phase agents often generate a second real-time plan and execute that instead.
- Run artifacts are now isolated by run root, but the system still lacks a single structured record of what each phase actually consumed and produced.
- Reviewer outcomes and downstream reporting still depend partly on free text, which weakens traceability.

This change is cross-cutting because it touches intake, planning, reviewer logic, reporting, and artifact conventions.

## Goals / Non-Goals

**Goals:**
- Make authoritative state ownership explicit for each workflow phase.
- Preserve stable logical thread identity while keeping every run artifact-isolated.
- Persist executed phase plans, reviewer outputs, and run manifests as structured artifacts.
- Make downstream agents prefer authoritative structured artifacts over stale mirrors or markdown-only inference.
- Reduce ambiguity between advisory planning and executed plans without changing the overall workflow order.

**Non-Goals:**
- Fully remove all legacy mirrors in one iteration.
- Redesign the reader → planner → reviewer → summarizer workflow sequence.
- Replace markdown reports with a UI-native reporting system.
- Introduce a new orchestration runtime or external persistence dependency.

## Decisions

### 1. `target_spec` remains the only authoritative target contract

`target_spec` SHALL remain the only contract that downstream logic is allowed to trust for target columns, target mode, submission schema, and evaluation semantics. Legacy mirrors may remain for compatibility, but they must be derived from `target_spec` or used only in narrowly scoped fallback code.

Alternative considered:
- Remove all legacy fields immediately.
  Rejected because too many downstream modules and tests still depend on compatibility mirrors.

### 2. `thread_id` and `run_id` SHALL model different identities

`thread_id` SHALL represent the stable logical workflow/thread identity used for resume, checkpointing, and observability grouping. `run_id` SHALL represent a unique execution instance used for per-run artifact roots and manifests.

Alternative considered:
- Keep overloading `thread_id` as the unique run identifier.
  Rejected because it breaks continuity for resumable workflows and makes tracing fragment by construction.

### 3. `execution_plan` owns phase plans; executed plans must be persisted separately

The phase-level `planner` will continue to own orchestration intent and success criteria, and it SHALL author the executable plan for the current phase. Phase agents SHALL consume that planner-authored phase plan on the first attempt and SHALL persist the actual executed plan, including whether it came from the planner output or from a deterministic fallback when the planner bundle is missing or invalid.

Alternative considered:
- Let phase agents regenerate their own phase-local plans.
  Rejected because it keeps the most important handoff implicit and makes phase execution drift from the shared workflow contract.

### 4. Every phase boundary should have compact structured artifacts

Each important phase SHALL be inspectable from artifacts without replaying logs:

- `run_manifest.json`: run-level snapshot
- phase executed-plan artifact
- phase review artifact

This keeps technical reports and business reports human-friendly while making structured debugging primary.

Alternative considered:
- Continue relying on logs plus markdown reports.
  Rejected because that makes machine validation and failure triage too fragile.

### 5. Downstream consumers should prefer structured artifacts over markdown text

`summarizer` and `business_translator` SHALL consume structured state and artifacts first. Markdown reports remain presentation outputs, not primary machine contracts.

Alternative considered:
- Let downstream consumers continue mining previous markdown reports for missing context.
  Rejected because report text is lossy and can drift from authoritative state.

## Risks / Trade-offs

- [Migration drift] → During transition, some modules may still read legacy mirrors; mitigate by documenting per-field authority and keeping fallback behavior explicit.
- [More artifacts per run] → Artifact count increases; mitigate by keeping each artifact compact and JSON-first.
- [Fallbacks can hide planning regressions] → Deterministic fallback remains necessary for resilience; mitigate by persisting plan source and fallback reason in the executed-plan artifact and reviewer checks.
- [Test churn] → Existing tests may encode old `thread_id` or artifact expectations; mitigate with focused schema and workflow fixture updates.

## Migration Plan

1. Add or refine schema fields and artifact keys for `run_id`, run manifests, executed-plan artifacts, and review artifacts.
2. Update intake and observability so `thread_id` remains stable while `run_id` drives artifact isolation.
3. Update planner and phase agents so the planner authors the current phase plan, phase agents execute that plan directly, and executed-plan provenance is recorded per phase.
4. Update reviewer, summarizer, and business translator to consume the new structured artifacts first.
5. Re-run representative tasks and then secondary smoke tasks to verify no regression in task success.

Rollback strategy:
- Keep compatibility fallbacks in place while introducing the new artifacts and authority rules.
- If a downstream consumer fails on the new structured artifact path, fall back to existing state fields while preserving the new artifacts for inspection.

## Open Questions

- Do we want a single shared schema for phase review artifacts, or separate phase-specific review payloads with a shared envelope?
