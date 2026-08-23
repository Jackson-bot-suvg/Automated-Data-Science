## Context

当前入口阶段已经初步支持多目标识别，但 target 相关信息仍然分散在
`TaskConfig`、`ProblemUnderstanding` 和 `DataState` 三处保存。Reader 和
Background Reviewer 需要在多个字段之间做回退、拼接和规范化，导致状态职责
混乱，也让单目标与多目标场景缺少统一的审核协议。

这次设计只覆盖三块：schema、Reader、Background Reviewer。目标不是一次性改完
整条 workflow，而是先在入口层建立一个清晰、可迁移的 target contract。下游
Planner、EDA、建模、submission、summary 和 observability 暂不切换到新协议。

## Goals / Non-Goals

**Goals:**

- 在 `GraphState` 中引入 `target_spec`，统一表达 target 解析结果、提交协议、
  评估提示和证据。
- 让 Reader 成为 `target_spec` 的唯一生产者，并把单目标与多目标解析收敛到同一
  套协议中。
- 让 Background Reviewer 只基于 `target_spec` 审核 target 是否已解析、是否泄漏、
  以及 submission schema 是否匹配。
- 保持旧 target 字段的短期兼容，降低对未迁移下游阶段的破坏风险。

**Non-Goals:**

- 不修改 Planner、EDA、Data Cleaning、Feature Engineering、Modeling、
  Submission、Summarizer 或 Observability 的消费逻辑。
- 不在本 change 内删除全部旧 target 字段或完成全链路迁移。
- 不扩展新的建模范式支持，例如 multi-output 训练策略或 submission 生成逻辑。

## Decisions

### 1. `target_spec` 作为唯一正式 target 协议，旧字段仅用于兼容

`GraphState` 将新增 `target_spec`，由它承载：

- target 解析状态，如 `status`、`target_mode`、`task_family`
- 用户 hint，如 `user_target_hint`、`user_task_family_hint`
- 权威 target 列，如 `target_columns`
- submission 协议，如 `id_columns`、`submission_columns`、`output_columns`
- evaluation 提示，如 `metric_hint`、`primary_metric`、`candidate_metrics`
- 证据链，如 `evidence`

同时保留旧字段一段时间，但职责改为兼容层，而不是主协议。这样做的原因是：

- 避免 target 语义继续分散，符合 DRY
- 让 Reader 和 Reviewer 不再依赖多处 fallback，符合 KISS
- 为后续下游渐进迁移保留缓冲，不需要一次性重写整条链路

备选方案：

- 直接删除所有旧字段并强制全链路切换：被拒绝，因为当前下游仍大量依赖
  `target_column`
- 新增一个过渡用 `target_config`：被拒绝，因为会制造第四份 target 状态，进一步恶化重复

### 2. Reader 负责生成完整 `target_spec`，而不是分别写多个 target 字段

Reader 将继续使用当前的结构化证据优先策略，但输出从“多处写 target_column /
target_columns”切换为“构造完整 `target_spec`”。

Reader 的解析顺序保持：

1. 用户显式目标输入或可直接映射的旧兼容字段
2. `train/test` 列差异与 `sample_submission` 交集
3. `overview` / `data_description` 中可验证的说明
4. 自然语言 hint 的兜底匹配

Reader 需要一次性填好：

- `status`
- `target_columns`
- `target_mode`
- `task_family`
- `prediction_kind`
- `submission`
- `evaluation`
- `evidence`

这样做的原因是把“推断 target”和“描述 target 协议”合并到一个职责边界内，符合 SRP。

备选方案：

- 让 Reader 只填 target 列名，submission/evaluation 继续由 reviewer 或下游补全：
  被拒绝，因为会重新引入职责分散

### 3. Background Reviewer 只审核 `target_spec`，不再依赖旧 target fallback 链

Background Reviewer 的主要判断将改为：

- `target_spec.status` 是否为 resolved
- `target_spec.target_columns` 是否全部存在于 train
- test 中是否包含任何 target columns
- 若存在 sample submission，其列集合与顺序是否匹配
  `target_spec.submission.submission_columns`

Reviewer 仍可在兼容期读取旧字段作为告警或回填依据，但审核结论只以 `target_spec`
为准。这样可以把 reviewer 收敛成“协议校验者”，而不是继续承担 target 拼装职责。

备选方案：

- 保留现有 reviewer 逻辑，仅追加 `target_spec` 校验：被拒绝，因为会保留双重真相

### 4. 采用“双写兼容、单读迁移”的入口迁移策略

本次 change 的迁移策略是：

1. schema 中新增 `target_spec`
2. Reader 生成 `target_spec`
3. Background Reviewer 只读 `target_spec`
4. 旧字段暂时保留，必要时由 Reader 同步写入，供未迁移下游继续工作

这比一次性全删旧字段更稳，也符合 YAGNI：先解决当前最核心的协议分散问题，再决定
下游何时切换。

## Risks / Trade-offs

- [Risk] 新旧字段并存一段时间会让状态看起来更复杂 → Mitigation: 在 design 和 spec 中明确 `target_spec` 是唯一正式协议，旧字段只作兼容
- [Risk] Reader 需要补全 submission/evaluation/evidence，设计复杂度高于当前实现 → Mitigation: 优先使用已有结构化证据，保持字段最小可用，不为未知场景过度设计
- [Risk] 下游尚未迁移，可能继续消费旧字段 → Mitigation: Reader 在兼容期继续双写旧字段，避免本 change 扩散
- [Risk] 某些数据集没有 sample submission，submission 协议可能不完整 → Mitigation: 允许保留空 submission 字段，但必须记录 unresolved evidence 或 warning

## Migration Plan

1. 扩展 `state_schema`，新增 `TargetSpec`、`SubmissionSpec`、`EvaluationSpec`
   和 `TargetEvidence`
2. 调整 Reader，把 target 解析结果统一写入 `target_spec`，并在兼容期同步旧字段
3. 调整 Background Reviewer，改为以 `target_spec` 为唯一审核依据
4. 增加针对单目标、多目标、target 未解析、test 含 target、sample submission
   不匹配等场景的单元测试

回滚方式：回退本 change 即可，因本次不涉及持久化存量数据迁移，也不改变外部接口。

## Open Questions

- 兼容期内，是否需要在旧字段与 `target_spec` 不一致时主动写 warning，以便尽早发现下游残留依赖
