## Why

The workflow now reaches technical and business reports more reliably, but the context handoff between agents is still too implicit. Target ownership is mirrored across multiple state buckets, phase-level plans can diverge from the plans actually executed inside each agent, and run artifacts are more observable than before but still not treated as the primary debugging contract.

## What Changes

- Define authoritative state ownership for each workflow phase and constrain legacy mirrors to compatibility-only usage.
- Separate logical workflow identity (`thread_id`) from per-run artifact identity (`run_id`) and make the distinction explicit in requirements.
- Make executed phase plans, reviewer outcomes, and run manifests first-class structured artifacts instead of relying on logs or markdown inference.
- Align planner behavior with phase-agent execution so every phase records whether it followed the planner plan, used a fallback plan, and what downstream reviewers should validate.
- Tighten downstream consumption so summarization and business translation prefer authoritative structured state and artifacts over stale mirrors or report-text inference.

## Capabilities

### New Capabilities
- `workflow-context-handoff`: Define authoritative state ownership, structured handoff artifacts, and downstream consumption rules across the multi-agent workflow.

### Modified Capabilities
- `llm-planner`: Clarify how phase-level planning relates to phase-internal execution plans and require executed-plan traceability.
- `main-cli-task-intake`: Clarify stable logical thread identity, unique run identity, and per-run artifact-root behavior.

## Impact

- Affected code: `app/intake/`, `app/agents/`, `app/tools/reviewer_tools.py`, `app/tools/`, `app/graph/`, `app/schemas/`, `run_multiagent.py`, `main.py`
- Affected artifacts: per-run manifests, executed-plan artifacts, review artifacts, report inputs
- Affected tests: task intake, planner, reviewer, summarizer, business translator, representative-task workflow tests
