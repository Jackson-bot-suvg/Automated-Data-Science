## Why

Multi-target datasets such as `data/plate_defect` expose several label columns in
`train.csv` and `sample_submission.csv`, but the current entry flow still treats
target understanding as a single-column problem. This causes Reader to misinfer
feature columns as targets and prevents the workflow from handing correct
multi-target intent to downstream stages.

## What Changes

- Update the CLI intake and confirmation flow to carry multi-target task
  understanding instead of assuming a single target column.
- Update Reader target inference to recognize multiple target columns from
  structured dataset evidence before falling back to fuzzy natural-language
  matching.
- Update Background Reviewer to validate approved Reader output when the task
  contains multiple target columns and to retry only when target scope remains
  unresolved.
- Extend the entry-flow state contract and docs so downstream phases can consume
  explicit multi-target metadata.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `main-cli-task-intake`: change task intake, Reader, reviewer, and confirmation
  requirements so the entry flow can infer, preserve, validate, and display
  multiple target columns.

## Impact

- Affected code: `main.py`, `app/intake/task_intake.py`, `app/agents/reader.py`,
  `app/agents/bg_reviewer.py`, and `app/schemas/state_schema.py`
- Affected docs: `docs/task_intake_layer.md`, `docs/reader.md`, and related CLI
  usage guidance
- Affected tests: CLI, Reader, and Background Reviewer unit coverage for
  multi-target datasets
