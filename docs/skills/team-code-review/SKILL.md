---
name: team-code-review
description: Review code changes and pull requests for this MultiAgent repository. Use when checking correctness, maintainability, security, performance, and compliance with docs/engineering-specification.md, docs/architecture.md, and docs/git.md. Apply a repeatable checklist covering code style, GraphState ownership, reserved artifact paths, tests, and release-readiness.
---

# Team Code Review

Read `docs/engineering-specification.md`, `docs/architecture.md`, and `docs/git.md` before reviewing unfamiliar modules.

Treat review as a defect-finding task first. Prioritize correctness, regressions, contract violations, and missing tests over style-only comments.

## Review Workflow

1. Scope the change.
   - Identify touched packages and map them to their ownership boundaries:
     - `app/graph/`: orchestration, routing, graph-facing adapters
     - `app/intake/`: CLI input normalization into existing `task_config` fields
     - `app/agents/`: business node entry points
     - `app/tools/`: execution and artifact helpers
     - `app/prompts/`: prompt text and metadata
     - `app/schemas/`: state, artifact, and error contracts
     - `tests/`: behavioral verification
     - `docs/`: architecture and engineering constraints

2. Check architecture and contract compliance.
   - Reject changes that add unauthorized top-level `GraphState` keys.
   - Confirm natural-language task intake keeps raw task, dataset root, and target hints under `task_config`.
   - Reject changes that rename reserved artifact keys or reserved artifact paths.
   - Reject changes that bypass reviewer gates through direct state mutation.
   - Reject changes that mutate fields owned by unrelated modules.
   - Reject changes that store non-serializable objects in checkpointed state.
   - Confirm node outputs remain partial state updates compatible with `GraphState`.
   - Confirm `main.py` sets `runtime_state.thread_id` and passes `configurable.thread_id` when invoking LangGraph.

3. Check code style and maintainability.
   - Require UTF-8 without BOM.
   - Use 4-space indentation.
   - Use `snake_case` for functions, variables, and modules.
   - Use `PascalCase` for classes and `UPPER_CASE` for constants.
   - Group imports in this order: standard library, third-party, local application imports.
   - Remove unused imports and avoid wildcard imports.
   - Keep module docstrings and public function docstrings concise and accurate.
   - Keep public interfaces typed.
   - Prefer small, single-purpose functions and modules.
   - Remove duplicated logic instead of copying and tweaking code.
   - Reject temporary `print` debugging in committed code.

4. Run security checks.
   - If database access is introduced, require parameterized queries. Never allow string-built SQL.
   - If shell commands are built, reject direct interpolation of untrusted input.
   - Validate file paths before reading, writing, or executing against them.
   - Do not allow hard-coded credentials, tokens, or private dataset secrets.
   - Require log desensitization for tokens, keys, PII, and sensitive business fields.
   - Confirm permission-sensitive operations are explicit and bounded.

5. Run performance checks.
   - Reject avoidable nested loops over large datasets when vectorized or aggregated operations are available.
   - Reject repeated full-data scans, repeated full copies, or repeated expensive artifact reads without justification.
   - If persistence or database access is added, check for N+1 query patterns.
   - If expensive computation is reused, prefer clear caching or memoization with explicit invalidation rules.
   - Do not add speculative caching for one-off work. Follow YAGNI.

6. Check testing and delivery gates.
   - Confirm changed behavior has at least one relevant unit test or integration smoke test.
   - If workflow, routing, or state contracts change, require integration coverage.
   - Confirm the branch still respects `docs/git.md`: no direct push to `main`, tests pass, lint/static checks pass, and DoD/PR expectations remain satisfiable.

## Review Output

Report findings in this order:

1. Findings
   - List defects first, ordered by severity.
   - Include file path, line reference, impact, and the reason it violates project rules.

2. Open questions
   - Note missing context only when it blocks a confident conclusion.

3. Residual risks
   - Call out untested paths, missing staging proof, or assumptions.

If no findings are discovered, say that explicitly and still mention any testing gaps or residual risks.

## Principle Checklist

- KISS: Prefer the simplest change that satisfies the requirement.
- YAGNI: Reject speculative abstractions, flags, and extension points with no current caller.
- DRY: Consolidate repeated validation, path handling, and state-update patterns.
- SOLID:
  - SRP: Each module should have one clear responsibility.
  - OCP: Extend behavior by adding focused modules instead of rewriting unrelated ones.
  - LSP: Preserve public contract behavior when replacing or extending implementations.
  - ISP: Keep interfaces narrow and task-specific.
  - DIP: Depend on contracts and shared schemas, not ad hoc concrete coupling.
