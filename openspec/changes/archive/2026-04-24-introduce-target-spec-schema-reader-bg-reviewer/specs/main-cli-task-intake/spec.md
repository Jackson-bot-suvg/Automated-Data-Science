## ADDED Requirements

### Requirement: 入口状态必须使用 target_spec 作为唯一正式 target 协议
系统 MUST 在入口阶段以 `state.target_spec` 统一表达 target 解析结果、提交协议、评估提示和证据；旧 target 字段如果存在，只能作为短期兼容镜像，不能再作为正式协议来源。

#### Scenario: Reader 解析出单目标
- **WHEN** Reader 在训练数据中解析出一个明确目标列
- **THEN** 系统必须将 `state.target_spec.status` 设为 `resolved`
- **THEN** 系统必须将 `state.target_spec.target_columns` 设为仅包含该目标列的列表
- **THEN** 系统必须将 `state.target_spec.target_mode` 标记为单目标语义
- **THEN** 任何旧的 `target_column` 或 `target_columns` 字段如果被保留，都只能反映 `target_spec` 的结果而不能覆盖它

#### Scenario: Reader 解析出多目标
- **WHEN** Reader 在训练数据、测试数据差异或 sample submission 中解析出多个目标列
- **THEN** 系统必须在 `state.target_spec.target_columns` 中保留完整且有序的目标列集合
- **THEN** 系统必须将 `state.target_spec.target_mode` 标记为非单目标语义
- **THEN** 系统必须在 sample submission 可用时填充 `state.target_spec.submission.output_columns`
- **THEN** 系统必须在 sample submission 可用时填充 `state.target_spec.submission.submission_columns` 作为最终提交列顺序

### Requirement: Background Reviewer 必须基于 target_spec 审核入口 target 一致性
Background Reviewer MUST 使用 `state.target_spec` 作为唯一审核依据，并在进入 Planner 前验证 target 是否已解析、训练集和测试集是否满足 target 契约，以及 sample submission 是否匹配 submission 协议。

#### Scenario: target_spec 一致且可通过
- **WHEN** `state.target_spec.status` 为 `resolved` 且所有 `target_spec.target_columns` 都存在于训练集
- **THEN** Background Reviewer 必须将 target 视为已明确
- **THEN** 如果测试集不包含任何 target columns，Background Reviewer 必须不得仅因目标数量大于 1 而拒绝通过
- **THEN** 如果 sample submission 存在且与 `target_spec.submission.submission_columns` 一致，Background Reviewer 必须允许 entry flow 继续

#### Scenario: 存在 target 泄漏或 submission schema 不匹配
- **WHEN** 测试集包含任一 `target_spec.target_columns` 或 sample submission 列顺序与 `target_spec.submission.submission_columns` 不一致
- **THEN** Background Reviewer 必须不得返回 approved
- **THEN** Background Reviewer 必须记录明确的 target 泄漏或 schema mismatch 原因
- **THEN** Background Reviewer 必须返回可阻止继续进入 Planner 的审核结论

## MODIFIED Requirements

### Requirement: Reader 根据数据集目录进行受控 ReAct 理解
Reader MUST 能根据 dataset root 读取目录、识别数据文件、读取有界元数据，并生成任务理解与完整 `target_spec`。

#### Scenario: 标准 Kaggle 风格目录
- **WHEN** 数据集目录包含 `train.csv`、`test.csv` 和 `sample_submission.csv`
- **THEN** Reader 必须识别训练集、测试集和提交样例路径
- **THEN** Reader 必须填充 `data_state.raw_train_path`、`data_state.raw_test_path` 和相关 `dataset_paths`
- **THEN** Reader 必须基于有界 CSV 元数据推断 `target_spec`、task type 和 candidate metrics

#### Scenario: 目录中存在说明文件
- **WHEN** 数据集目录包含 `overview.txt` 或 `data_description.txt`
- **THEN** Reader 可以读取有限长度的说明文本作为任务理解上下文
- **THEN** Reader 不得把完整大文件直接放入 LLM prompt
- **THEN** Reader 如果使用说明文本辅助 target 解析，必须把该来源记录到 `target_spec.evidence`

#### Scenario: 目录中可可靠识别多目标
- **WHEN** sample submission 的输出列与 train/test 列差异可可靠映射到多个目标列
- **THEN** Reader 必须保留完整目标列集合到 `target_spec.target_columns`
- **THEN** Reader 必须不得把该任务压缩成单个 `target_column`
- **THEN** Reader 必须生成与该目标集合一致的 `target_spec.submission` 协议

### Requirement: target 不确定时允许用户自然语言补充
系统 MUST 在 target 无法可靠推断时，通过 CLI 直接让用户补充自然语言目标，并重新执行 Reader 理解和 Background Reviewer 审核。

#### Scenario: Reader 无法确定目标列
- **WHEN** Reader 无法从 train/test 列差异、任务文本、目标提示或 sample submission 中生成 `status=resolved` 的 `target_spec`
- **THEN** Background Reviewer 必须返回 retry
- **THEN** CLI 必须提示用户用自然语言说明想预测的目标
- **THEN** 系统必须将用户补充内容传回 Reader 重新理解

#### Scenario: 自然语言补充仍无法确定
- **WHEN** 用户补充自然语言目标后 Reader 仍无法映射到具体列名或具体目标列集合
- **THEN** CLI 必须展示候选列
- **THEN** 用户必须能输入精确列名或精确目标列集合以继续
