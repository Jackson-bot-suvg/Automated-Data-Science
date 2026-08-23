## Context

当前自然语言 CLI 入口把目标理解建模成单个 `target_column`，并且 Reader
会在结构化证据不足或尚未充分使用时，过早对任务文本做模糊列名匹配。像
`data/plate_defect` 这样的数据集会在 `train.csv`、`test.csv` 和
`sample_submission.csv` 中同时暴露 7 个目标列，但现有入口流程既无法稳定
识别它们，也无法把完整目标范围传给后续阶段。

这次设计只覆盖团队分工中的入口段：`main.py`、TaskIntake、Reader、
Background Reviewer，以及它们依赖的状态契约与文档。下游规划、建模和
提交链路视为已经具备消费多目标元数据的能力，本设计不对其行为做降级提示
或兼容兜底。

## Goals / Non-Goals

**Goals:**

- 让入口流程能明确表达单目标和多目标两种 target scope。
- 让 Reader 在多目标数据集上优先依据结构化证据推断目标列集合。
- 让 Background Reviewer 能审核多目标 Reader 输出并在 target scope
  明确时放行。
- 让 CLI 在确认与补充交互中展示和传递多目标理解结果。

**Non-Goals:**

- 不修改 Planner、特征工程、建模、submission 或后置 reviewer 的具体行为。
- 不为所有 multi-output 机器学习范式设计统一语义，只覆盖当前入口识别与传递。
- 不改变数据集目录发现规则，也不扩大 Reader 的文件读取范围。

## Decisions

### 1. 在入口状态契约中增加 plural target 字段

入口相关状态将新增 `target_columns: list[str]`，至少覆盖
`task_config`、`problem_understanding` 和 `data_state` 中 Reader 与
Background Reviewer 会读取或写入的 target 信息。

- 单目标任务：同时写入 `target_column` 和 `target_columns=[target_column]`
- 多目标任务：写入完整 `target_columns`，并将 `target_column` 视为非权威字段

这样做的原因是：

- 保持单目标路径简单，符合 KISS
- 避免用逗号拼接字符串表达多目标，减少解析歧义，符合 DRY
- 为团队已完成的下游多目标支持提供明确契约

备选方案：

- 只复用 `target_column`，用逗号拼接多个列名：被拒绝，因为会让校验、展示和
  序列化都依赖重复解析逻辑
- 多目标时把 `target_column` 设为第一个目标列：被拒绝，因为会重新引入错误的
  单目标假设

### 2. Reader 采用“结构化证据优先、模糊匹配兜底”的推断顺序

Reader 的 target inference 会调整为：

1. 精确的显式 target 输入
2. `sample_submission.csv` 非 ID 输出列与 `train/test` 差异列的交集
3. `overview.txt` / `data_description.txt` 中可验证的多目标说明
4. 仅在以上都不能可靠确定时，才使用自然语言 hint 与任务文本的模糊匹配

关键规则：

- 一旦结构化证据可靠识别出多个目标列，Reader 不再继续尝试把任务文本压缩成单个目标列
- 模糊匹配只能作为兜底，不得覆盖更强的结构化证据

这样做可以直接消除 `plate_defect -> Steel_Plate_Thickness` 这类误判。

备选方案：

- 保持现有顺序，只在末尾补一个多目标分支：被拒绝，因为误匹配会在更早阶段发生

### 3. Background Reviewer 以“target scope 是否明确”为准，而不是“是否只有一个 target”

Background Reviewer 会通过统一 helper 读取 resolved target scope：

- 若存在合法的 `target_columns`，且全部在训练集列中，则认为 target scope 明确
- 若只有合法的单个 `target_column`，同样视为明确
- 只有在单复数字段都无法可靠解析时，才返回 `retry`

这样做把 reviewer 的职责收敛为“验证 Reader 输出是否完整可信”，而不是替
下游能力做限制判断，符合 SRP。

备选方案：

- 继续只检查 `target_column`：被拒绝，因为无法通过多目标输入

### 4. CLI 交互按 target scope 展示，不引导用户缩窄多目标

`main.py` 会基于 resolved target scope：

- 在确认界面显示 `Target column` 或 `Target columns`
- 仅当 target scope 未解析成功时，才向用户发起补充提问
- 当 Reader 已推断出多个目标列时，CLI 直接展示并继续确认，不要求用户再选单个列

这与团队协作边界保持一致：入口层负责正确表达任务，后续层负责消费该表达。

## Risks / Trade-offs

- [Risk] 状态 schema 变更会影响现有单元测试和文档 → Mitigation: 为单目标保留原字段行为，并同步更新 tests/docs
- [Risk] 某些 sample submission 可能包含非目标辅助列 → Mitigation: 仅接受与 train/test 差异一致的列，并允许 overview 文本交叉验证
- [Risk] 多目标与单目标字段并存可能让调用方混淆 → Mitigation: 在 docs 中明确 `target_columns` 为多目标权威字段，单目标时二者保持一致
- [Risk] 目标提示语法如果未来支持逗号分隔列表，当前解析逻辑可能需要再次扩展 → Mitigation: 本次先保持 hint 解析最小化，不提前实现未确认语法

## Migration Plan

1. 扩展入口相关 schema 与辅助读取逻辑，支持 plural target 字段
2. 调整 Reader 推断顺序和返回 payload，补充多目标识别
3. 调整 Background Reviewer 校验逻辑和 CLI 展示逻辑
4. 补齐针对 `plate_defect` 风格数据集的测试与文档

回滚方式：整体回退本 change 对入口层和 schema 的修改即可，不涉及数据迁移或持久化格式迁移。

## Open Questions

- `--target` 是否需要在本 change 内支持逗号分隔的精确多列输入，还是先依赖自动推断与现有自然语言补充流程
