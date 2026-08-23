---
name: create-module
description: Create or scaffold a new module, phase, or feature slice for this MultiAgent repository. Use when adding an agent, graph component, schema, tool, prompt, or related tests while staying compliant with docs/engineering-specification.md and docs/architecture.md. Follow the repository structure, GraphState boundaries, artifact naming rules, and minimal-change principles.
---

# Create Module

Read `docs/engineering-specification.md` and `docs/architecture.md` before creating files.

Create the smallest module set that satisfies the request. Do not import web-controller, service-layer, or DTO conventions from other projects unless the repository explicitly adopts them.

## Module Selection

Choose the target package by responsibility:

- `app/agents/`: business node entry points such as `run_reader` or `run_reviewer`
- `app/intake/`: CLI input normalization before Reader, including natural-language task, dataset root, and target hints stored under `task_config`
- `app/graph/`: workflow construction, routing, and graph-facing state adapters
- `app/llm/`: unified LLM calling service layer, multi-provider routing, provider configuration, agent LLM configuration (`get_agent_config`), and high-level helpers (`complete_json`, `render_text`)
- `app/tools/`: execution helpers and artifact utilities
- `app/prompts/`: prompt templates and prompt metadata
- `app/schemas/`: state, artifact, IO, and error contracts
- `tests/unit/`: focused unit coverage
- `tests/integration/`: workflow and contract smoke coverage
- `docs/`: architecture or contract updates only when rules actually change

## Required Workflow

1. Define the module responsibility.
   - State the single responsibility in one sentence.
   - Identify which existing state fields the module may read and write.
   - Reuse existing top-level `GraphState` keys whenever possible.
   - Do not add new top-level keys without architecture approval.

2. Choose the minimum required files.
   - For a new agent phase, usually create or update:
     - `app/agents/<phase>.py`
     - `app/prompts/<phase>_prompt.py` if prompt text is needed
     - `app/graph/workflow.py` and `app/graph/routes.py` only if routing changes
     - relevant tests under `tests/unit/` or `tests/integration/`
   - For a new tool helper, usually create:
     - `app/tools/<name>.py`
     - focused unit tests
   - For CLI intake behavior, usually update:
     - `app/intake/task_intake.py`
     - `main.py` only if the user-facing flow changes
     - focused tests for TaskIntake and CLI confirmation/retry behavior
   - For a schema or contract update, change:
     - `app/schemas/*.py`
     - matching docs only when the documented contract changes

3. Apply file-level rules.
   - Add a concise module docstring.
   - Keep imports minimal and ordered.
   - Add type hints for public functions.
   - Keep each file focused on one responsibility.
   - Do not edit unrelated files.

4. Respect node contract rules for business phases.
   - Public entry function shape must remain `run_<phase>(state: GraphState) -> dict`.
   - Return partial state updates, not ad hoc objects.
   - Use only serializable values in returned state.
   - Preserve append-only semantics for `phase_history`, `review_history`, `artifacts.figures`, and `messages`.
   - Use reserved artifact paths instead of inventing new names.

5. Add the minimum test coverage.
   - Add at least one unit test or contract smoke test for new behavior.
   - If state schema, routing, or workflow order changes, update or add integration coverage.
   - Keep tests targeted and readable.

## Minimal Templates

Use this template for a new agent module:

```python
"""<module summary>."""

from __future__ import annotations

from app.schemas.state_schema import GraphState


async def run_<phase>(state: GraphState) -> dict:
    """Execute the <phase> phase and return partial state updates."""
    raise NotImplementedError("<reason if scaffold only>")
```

When the agent needs LLM calls, use the helpers from `app/llm/`:

```python
from app.llm import complete_json, render_text, get_agent_config

# For JSON-structured LLM responses (most agents)
payload = await complete_json(
    agent_name="<phase>",
    system_prompt=SYSTEM_PROMPT,
    user_prompt=build_prompt(...),
    fallback=default_payload,
)

# For free-text LLM responses
text = await render_text(
    agent_name="<phase>",
    system_prompt=SYSTEM_PROMPT,
    user_prompt=build_prompt(...),
    fallback="default text",
)
```

When a phase module needs structured planning via `BasePlanner`, bind the planner to the phase's own `AGENT_<NAME>_*` config segment instead of hardcoding planner-wide config:

```python
from app.planners.base import BasePlanner

planner = BasePlanner(agent_name="<phase>")
plan = await planner.plan_data_cleaning(profile_json)
```

Use this template for a focused test:

```python
def test_<behavior>() -> None:
    state = GraphState()

    result = run_<phase>(state)

    assert "<key>" in result
```

## Definition of Done

Only treat the module as complete when all conditions hold:

- file is placed in the correct package
- public entrypoint exists
- ownership of state reads and writes is clear
- reserved artifact names remain intact
- at least one relevant test exists
- no unrelated module responsibilities leaked into the change

## Principle Checklist

- KISS: Start with the smallest workable scaffold.
- YAGNI: Do not pre-build extension hooks, extra layers, or unused config.
- DRY: Reuse existing schema, artifact, and helper patterns before creating new ones.
- SOLID:
  - SRP: Keep each module and function narrowly scoped.
  - OCP: Extend by adding targeted modules instead of rewriting stable contracts.
  - ISP: Keep public APIs small and explicit.
  - DIP: Depend on shared schemas and contracts, not on hidden cross-module knowledge.
