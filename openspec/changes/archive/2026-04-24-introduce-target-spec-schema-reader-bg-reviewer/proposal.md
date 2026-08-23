## Why

当前 target 语义分散在 `TaskConfig`、`ProblemUnderstanding` 和 `DataState` 中重复保存，Reader 和 Background Reviewer 需要在多个字段之间回退和拼接，导致单目标与多目标任务都缺少统一协议。现在需要把 target 解析与校验收敛为一个明确的状态契约，先统一 schema、Reader 和 Background Reviewer，为后续下游阶段迁移打基础。

## What Changes

- 在 `GraphState` 中新增 `target_spec`，作为 target 定义、提交协议、评估提示和证据记录的统一状态对象。
- 调整 `TaskConfig`、`ProblemUnderstanding` 和 `DataState` 的职责边界，停止把 target 信息作为正式长期协议分散存放；旧字段仅保留为短期兼容字段。
- 更新 Reader，使其成为 `target_spec` 的唯一生产者，负责根据用户 hint、train/test 差异、sample submission 和 overview 生成完整 target 协议。
- 更新 Background Reviewer，使其基于 `target_spec` 审核 target 是否 resolved、train/test 是否存在 target 泄漏、以及 sample submission 是否与 submission 协议一致。
- 为 schema、Reader 和 Background Reviewer 增加针对单目标、多目标和 target 未解析场景的测试覆盖。

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `main-cli-task-intake`: 调整 state contract、Reader target 解析规则和 Background Reviewer 审核规则，使 entry flow 以 `target_spec` 作为唯一 target 协议。

## Impact

- Affected code: `app/schemas/state_schema.py`, `app/agents/reader.py`, `app/agents/bg_reviewer.py`, 以及与它们直接耦合的 schema/contract helpers
- Affected tests: Reader、Background Reviewer、CLI entry flow 和 state schema 相关单元测试
- Affected behavior: entry flow 在进入下游阶段前，将以 `target_spec` 而不是分散旧字段表达 target 解析结果与审核依据
- Non-goals in this change: `Planner`、`EDA`、`Data Cleaning`、`Feature Engineering`、`Modeling`、`Submission`、`Summarizer`、`Observability`
