### Git Collaboration Specifications
The following branch management strategy must be adopted:

- main: Stable version

- dev: Integration branch

- feature/<module-name>: Personal feature branches

## 1 Merge Rules

Before any code is merged into dev, the following conditions must be met:

- Local unit tests pass

- At least 1 mock integration test passes

- The Definition of Done (DoD) must be filled in the PR

- CI automated tests (unit tests + integration tests) must pass

- Linting static checks must pass

- Test coverage reaches 80% or higher

- No modification of field names responsible for other team members

- The commit message clearly describes the changes made

## 2 Branch and PR Specifications

- Personal development branches must use feature/<module-name>; feat/<name> or fix/<name> are also allowed.

- Submitted PRs must be reviewed by other team members; team members shall take turns reviewing, and a PR can be merged after approval by one reviewer.

## 3 Prohibited Items

- Direct pushes to main are prohibited.

- Unauthorized modification of shared field names is prohibited.

- Modification of artifact path specifications is prohibited.

- Using temporary print statements instead of formal return values is prohibited.