## Why

Recent workflow failures show that the planner can still generate phase-valid JSON that is operationally poor: it may choose tools that are syntactically allowed but semantically unsafe for the observed schema, overuse empty plans, or miss task-specific scoring/modeling guidance. This directly reduces workflow stability even when the downstream phase agents are behaving correctly.

## What Changes

- Tighten planner prompt contracts so each execution phase receives only its own tool context plus stronger phase-specific hard rules.
- Add phase-specific tool usage guidance, anti-pattern guidance, and lightweight self-check requirements to reduce invalid or low-value plans.
- Require planner prompts to prefer a safe minimal executable plan over empty step lists for actionable phases.
- Strengthen model-building planning guidance so binary, multiclass, multilabel, and regression tasks get clearer candidate-model, scoring, and CV instructions.
- Keep changes narrowly scoped to planner prompting and planner-side plan validation; do not redesign phase agents or workflow routing.

## Capabilities

### New Capabilities

### Modified Capabilities
- `llm-planner`: Tighten phase-scoped tool planning behavior, add stronger prompt guardrails, and reduce invalid planner output for execution phases.

## Impact

- Affected code: `app/agents/planner.py`, `app/prompts/planner_prompt.py`
- Affected behavior: phase-level tool planning for `data_cleaning`, `indepth_eda`, `feature_engineering`, and `model_building`
- Affected tests: planner-focused unit tests and one representative workflow validation run
