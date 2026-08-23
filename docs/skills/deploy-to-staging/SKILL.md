---
name: deploy-to-staging
description: Run a guarded staging release workflow for this MultiAgent repository. Use when validating a branch for integration, preparing a staging deployment, or checking release readiness against docs/git.md and docs/engineering-specification.md. Follow preconditions, quality gates, deployment safety checks, and rollback-minded verification. Stop and report blockers when staging infrastructure is not configured.
---

# Deploy To Staging

Read `docs/git.md` and `docs/engineering-specification.md` before running deployment checks.

Treat this repository as deployment-constrained by default. If no staging pipeline, environment, or credentials are configured, stop early and report the blocker instead of improvising an ad hoc release path.

## Preconditions

Require all of the following before deployment:

- work from a feature branch, `feat/*`, or `fix/*`
- never deploy from a dirty tree without explicitly accounting for local changes
- staging credentials and required environment variables are available
- the target commit is known
- the intended scope and rollback owner are clear

## Workflow

1. Validate branch and release scope.
   - Confirm the branch is not `main`.
   - Summarize the feature or fix being promoted.
   - Note any schema, workflow, or artifact-path changes that raise deployment risk.

2. Run repository quality gates.
   - Run unit and integration tests when available.
   - Run lint or static checks if the repository provides them.
   - If no dedicated build step exists, at minimum run a syntax or import smoke check.
   - If a required gate cannot run, report it as an explicit blocker.

3. Re-check architecture constraints.
   - Confirm no unauthorized top-level `GraphState` keys were added.
   - Confirm reserved artifact paths were not renamed.
   - Confirm reviewer-gated phases still respect the documented flow.

4. Execute the project-approved deployment path only.
   - Prefer repository scripts, CI jobs, or documented commands.
   - Do not invent one-off deployment commands.
   - Do not bypass permission or environment checks.

5. Verify staging health.
   - Run the lightest reliable smoke test after deployment.
   - Verify the expected workflow can still initialize and checkpoint safely.
   - Check logs for startup errors, contract mismatches, or missing artifacts.

6. Prepare rollback context.
   - Record the deployed commit.
   - Record the previous known-good commit or release reference.
   - If rollback steps are unknown, report that as release risk.

## Suggested Commands

Use repository-supported commands when they exist. For the current scaffold repository, the default validation path is:

```powershell
pytest tests/unit tests/integration
python -m compileall app tests main.py
python main.py --task "predict customer churn" --dataset data/bank_churn --yes
```

If additional project-approved commands are later added, prefer those commands instead of expanding this list.

## Success Criteria

Treat deployment as successful only when:

- required tests pass
- required static checks pass or are explicitly not configured
- deployment command or CI job completes successfully
- smoke validation on staging passes
- rollback reference is captured

## Failure Criteria

Treat deployment as failed when any of the following occurs:

- missing staging credentials or environment variables
- test, lint, or smoke failure
- contract violation against the engineering specification
- deployment path is undocumented or improvised
- rollback path is unknown

## Reporting Format

Report the outcome in this order:

1. Preconditions
2. Executed checks and commands
3. Deployment result
4. Smoke verification result
5. Blockers or rollback notes

## Principle Checklist

- KISS: Follow the shortest approved release path.
- YAGNI: Do not add deployment machinery that the repository does not yet use.
- DRY: Reuse existing CI or script entrypoints instead of duplicating shell logic.
- SOLID: Keep validation, deployment, and rollback responsibilities explicit and separated.
