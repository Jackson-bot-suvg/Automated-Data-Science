## 1. Planner Prompt Guardrails

- [x] 1.1 Remove or de-emphasize inactive broad planner prompt paths so the active execution-phase prompt is the clear source of truth
- [x] 1.2 Tighten the active phase prompt with phase-specific hard rules and anti-pattern guidance for cleaning, in-depth EDA, feature engineering, and model building
- [x] 1.3 Replace the default “return empty steps when uncertain” guidance with “prefer a safe minimal executable plan when the phase is actionable”
- [x] 1.4 Add an internal self-check rubric to the active planner prompt without changing the JSON output schema

## 2. Model-Building Task Guidance

- [x] 2.1 Add explicit task-family guidance for binary classification, multiclass classification, multilabel classification, and regression in the model-building planner prompt
- [x] 2.2 Keep the guidance aligned with the existing planner allowlist and parameter schema so it does not introduce unsupported tools or fields

## 3. Planner-Side Validation

- [x] 3.1 Tighten planner-side plan validation where needed so schema-incompatible or protected-column-hostile parameters are more likely to be dropped before execution
- [x] 3.2 Keep the implementation scoped to planner prompting and planner-side normalization only; do not redesign phase-agent execution

## 4. Verification

- [x] 4.1 Add or update planner-focused tests for phase-scoped prompts, protected-column guidance, and task-aware model-building guidance
- [x] 4.2 Run one representative workflow task to verify the planner still emits executable phase plans after the prompt tightening
