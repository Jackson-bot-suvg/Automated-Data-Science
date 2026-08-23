## Context

The current planner already uses a phase-scoped allowlist and a ToolCallPlan-only output contract, but the prompt remains too generic. It tells the model which tools are available and what parameter schema to follow, yet it does not strongly encode when a tool must not be used, when an empty plan is unacceptable, or how task family should influence model-building decisions.

This gap shows up in real runs:

- feature engineering plans may target protected columns or choose tools that do not match the observed schema
- planner may return empty steps and fall back too often
- model-building plans may be schema-valid but still poorly matched to binary, multiclass, multilabel, or regression tasks

## Goals / Non-Goals

**Goals:**
- Keep planner visibility phase-scoped and make that constraint explicit in the prompt contract.
- Add phase-specific hard rules and anti-pattern guidance that reduce unsafe tool selection.
- Reduce empty-plan outcomes by preferring safe minimal executable plans for actionable phases.
- Add planner-side self-check guidance before JSON output.
- Improve task-type-specific guidance for model-building plans without changing model-training internals.

**Non-Goals:**
- Redesign phase agents or reviewer logic.
- Add new feature-engineering or modeling tools.
- Change graph routing, retry policy, or report structure.
- Introduce a large few-shot library or external prompt dependency.

## Decisions

### 1. Keep a single active prompt path and strengthen it instead of adding new planner modes

The implementation will continue to use the existing phase-toollist planning path for execution phases. We will improve the active prompt instead of reviving the older broader planning prompt shape.

Alternative considered:
- Reintroduce a richer phase-planning envelope and then extract tool steps from it.
  Rejected because it adds output surface area and increases structured-output failure risk.

### 2. Encode hard rules and anti-patterns directly in the prompt

Each phase prompt will include concise negative guidance such as:

- do not mutate target or ID columns
- do not choose regex tools for clearly non-string columns
- do not choose arithmetic interactions for non-numeric inputs
- do not choose unsupported or task-incompatible modeling configurations

Alternative considered:
- Rely only on downstream phase-agent protection logic.
  Rejected because that catches errors late and still wastes planning attempts.

### 3. Replace “empty plan when uncertain” with “safe minimal plan when actionable”

For actionable phases, the planner should prefer one or more conservative, executable steps rather than returning an empty plan. Empty plans remain acceptable only for non-actionable cases or when phase evidence clearly supports no operation.

Alternative considered:
- Keep empty plans as the default uncertainty behavior.
  Rejected because it encourages planner no-op behavior and pushes too much work into deterministic fallback.

### 4. Add internal self-check expectations without expanding the output schema

The prompt will ask the model to internally verify allowlist membership, schema validity, protected-column safety, and observed-column existence before emitting JSON, but the output remains the same compact `ToolCallPlan`.

Alternative considered:
- Add a visible checklist field to the JSON output.
  Rejected because it increases parsing surface area without helping runtime execution.

### 5. Add explicit task-family guidance for model building

The model-building phase prompt will include concise task-family guidance for:

- binary classification
- multiclass classification
- multilabel classification
- regression

This guidance will cover preferred scoring semantics, CV defaults, and candidate-model expectations without leaking into training implementation details.

Alternative considered:
- Encode all task-family behavior only in deterministic fallback.
  Rejected because the LLM planner should still be able to generate a good first-attempt plan.

## Risks / Trade-offs

- [Over-constraining prompts] → Too many rules can make the model conservative; mitigate by keeping hard rules short and phase-specific.
- [Prompt drift from actual tool behavior] → Guidance can become stale if tools change; mitigate by deriving examples and constraints from the current schema/allowlist only.
- [False confidence] → Better prompts reduce but do not eliminate bad plans; mitigate by preserving planner-side validation and deterministic fallback.
