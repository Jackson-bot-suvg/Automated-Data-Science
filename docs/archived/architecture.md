Multi-Agent Automated Data Science System

Architecture Overview

1. Goal

This repository is organized as a LangGraph-first project.
The graph orchestrates phase transitions, while agent modules, tools, prompts, and schemas stay isolated by responsibility.

2. High-Level Layers

- `app/graph/`: LangGraph orchestration entry points, routing, and graph-facing state adapters
- `app/agents/`: node-facing business modules and placeholder public entry points
- `app/llm/`: unified LLM calling interface, multi-provider routing, and configuration management
- `app/tools/`: execution helpers and artifact utilities
- `app/prompts/`: prompt templates and prompt metadata
- `app/schemas/`: shared contracts, state schema, artifact schema, and error schema
- `docs/`: architecture and engineering specifications
- `docs/skills/`: team-shared Codex skill definitions and workflow checklists
- `tests/`: unit and integration validation

3. Ownership Rules

- Graph routing belongs in `app/graph/routes.py`.
- Workflow construction belongs in `app/graph/workflow.py`.
- Shared state contracts belong in `app/schemas/state_schema.py`.
- Business node placeholders belong in `app/agents/`.
- No prompt, tool, or agent module may redefine top-level state keys.
- Model routing, provider configuration, provider implementations, agent LLM configuration, and high-level LLM helpers belong in `app/llm/`.
- Business nodes must call LLM through the service layer (`app/llm/`) rather than directly holding SDK clients.
- `app/services/` has been removed; all LLM calls are routed through `app/llm/`.

4. Implementation Status

All LLM interactions in the project are unified through `app/llm/`. Seven agents (reader, planner, summarizer, bg_reviewer, reviewer, data_cleaning, feature_engineering) use the service layer via `complete_json()`, `render_text()`, or direct `LLMService.chat()` calls. Each agent's model configuration is managed through `AGENT_<NAME>_*` environment variables via `get_agent_config()`.

5. Observability

- Observability integrations belong to infrastructure modules, not business nodes.
- LangSmith tracing must be wired through shared runtime entry points such as graph invocation and LLM planner wrappers.
- Tracing metadata may be derived from `GraphState` and runtime configuration, but tracing must not mutate `GraphState`.
- Business node contracts remain unchanged when tracing is enabled or disabled.
