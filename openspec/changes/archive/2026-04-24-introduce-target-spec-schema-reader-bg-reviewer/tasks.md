## 1. Schema Contract

- [x] 1.1 在 `app/schemas/state_schema.py` 中新增 `TargetSpec`、`SubmissionSpec`、`EvaluationSpec` 和 `TargetEvidence`
- [x] 1.2 在 `GraphState` 中加入 `target_spec`，并调整 `TaskConfig`、`ProblemUnderstanding`、`DataState` 中 target 相关字段的职责注释或兼容定位
- [x] 1.3 为新的 `target_spec` 默认值、单目标语义和多目标语义补充 schema 级测试

## 2. Reader TargetSpec Generation

- [x] 2.1 重构 Reader 的 target 解析逻辑，使其统一生成 `target_spec` 而不是以旧 target 字段作为主输出
- [x] 2.2 让 Reader 基于用户 hint、train/test 差异、sample submission 和 overview 填充 `target_spec.status`、`target_columns`、`target_mode`、`task_family`、`prediction_kind`
- [x] 2.3 让 Reader 填充 `target_spec.submission`、`target_spec.evaluation` 和 `target_spec.evidence`
- [x] 2.4 在兼容期内由 Reader 同步写入必要的旧 target 字段，保证未迁移下游仍可继续运行

## 3. Background Reviewer Validation

- [x] 3.1 重构 Background Reviewer，使其仅以 `target_spec` 作为 target 审核依据
- [x] 3.2 增加对 `target_spec.status`、训练集缺失 target、测试集含 target 和 sample submission schema mismatch 的审核规则
- [x] 3.3 调整 Background Reviewer 的审核详情与报错信息，使其明确输出 target unresolved、leakage 或 submission mismatch 原因

## 4. Verification

- [x] 4.1 为 Reader 增加单目标、多目标、overview 证据、sample submission 协议和 unresolved target 场景测试
- [x] 4.2 为 Background Reviewer 增加 resolved target、test 含 target、sample submission 不匹配和多目标通过场景测试
- [x] 4.3 运行与 schema、Reader、Background Reviewer 相关的单元测试，并确认 change 范围内行为通过
