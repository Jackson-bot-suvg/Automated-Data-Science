## ADDED Requirements

### Requirement: Reader 优先使用结构化证据识别多目标标签
系统必须在多目标数据集上优先使用结构化数据证据推断目标范围，而不是先用自然语言模糊匹配单个列名。

#### Scenario: sample submission 和 train/test 差异共同指向多个目标列
- **WHEN** `sample_submission.csv` 的非 ID 输出列与 `train.csv` 相比 `test.csv` 多出的列集合一致，并且数量大于 1
- **THEN** Reader MUST 将这些列识别为目标列集合
- **THEN** Reader MUST 保留完整目标列集合供后续阶段使用
- **THEN** Reader MUST NOT 在此情况下把任务文本模糊匹配到某个特征列并将其当作唯一目标

## MODIFIED Requirements

### Requirement: TaskIntake 保留原始任务和数据集根目录
系统必须有独立 TaskIntake 层负责把 CLI 输入转换为初始状态，并保留用户原始自然语言任务和数据集根目录。

#### Scenario: 数据集参数是目录
- **WHEN** `--dataset` 指向一个目录
- **THEN** TaskIntake 必须在现有顶层 `task_config` 内保存该目录作为 dataset root，或使用等价的现有字段承载该语义
- **THEN** TaskIntake 不得新增 `GraphState` 顶层字段
- **THEN** TaskIntake 不得把目录路径伪装成 `dataset_paths.train`
- **THEN** Reader 必须负责读取目录并判断实际数据文件

#### Scenario: 用户提供目标提示
- **WHEN** 用户传入 `--target` 且内容可能是精确列名、自然语言目标提示或与多目标任务相关的说明
- **THEN** TaskIntake 必须保留该目标提示
- **THEN** Reader 必须结合目标提示、sample submission、train/test 列差异和真实列名判断具体 target column 或 target columns
- **THEN** 当推断结果包含多个目标列时，系统必须保留完整目标列集合，而不是压缩成单个目标列

### Requirement: Reader 根据数据集目录进行受控 ReAct 理解
Reader 必须能根据 dataset root 读取目录、识别数据文件、读取有界元数据，并生成任务理解。

#### Scenario: 标准 Kaggle 风格目录
- **WHEN** 数据集目录包含 `train.csv`、`test.csv` 和 `sample_submission.csv`
- **THEN** Reader 必须识别训练集、测试集和提交样例路径
- **THEN** Reader 必须填充 `data_state.raw_train_path`、`data_state.raw_test_path` 和相关 `dataset_paths`
- **THEN** Reader 必须基于有界 CSV 元数据推断 target column 或 target columns、task type 和 candidate metrics
- **THEN** 当 sample submission 与 train/test 结构表明存在多个目标列时，Reader 必须输出完整目标列集合

#### Scenario: 目录中存在说明文件
- **WHEN** 数据集目录包含 `overview.txt` 或 `data_description.txt`
- **THEN** Reader 可以读取有限长度的说明文本作为任务理解上下文
- **THEN** Reader 不得把完整大文件直接放入 LLM prompt
- **THEN** 说明文本可以用于确认单目标或多目标任务语义，但不得覆盖更强的结构化数据证据

### Requirement: target 不确定时允许用户自然语言补充
系统必须在 target 无法可靠推断时，通过 CLI 直接让用户补充自然语言目标，并重新执行 Reader 理解和 Background Reviewer 审核。

#### Scenario: Reader 无法确定目标范围
- **WHEN** Reader 无法从 sample submission、train/test 列差异、任务文本或目标提示中确定 target column 或 target columns
- **THEN** Background Reviewer 必须返回 retry
- **THEN** CLI 必须提示用户用自然语言说明想预测的目标
- **THEN** 系统必须将用户补充内容传回 Reader 重新理解

#### Scenario: Reader 已可靠识别多个目标列
- **WHEN** Reader 能从结构化证据中可靠识别多个目标列
- **THEN** Background Reviewer 必须不得仅因为目标列数量大于 1 而返回 retry
- **THEN** CLI 必须保留并传递这些目标列到后续 workflow
- **THEN** CLI 不得强制要求用户把多目标任务收窄为单个目标列

#### Scenario: 自然语言补充仍无法确定
- **WHEN** 用户补充自然语言目标后 Reader 仍无法映射到具体列名或目标列集合
- **THEN** CLI 必须展示候选列
- **THEN** 用户必须能输入精确列名或精确目标列集合以继续

### Requirement: 进入 Planner 前必须有人类确认
系统必须在 Background Reviewer 审核通过后，将 Reader 理解结果展示给用户确认，确认后才进入 Planner。

#### Scenario: 用户确认理解结果
- **WHEN** Background Reviewer 返回 approved
- **THEN** CLI 必须展示任务摘要、任务类型、目标列或目标列集合、数据文件、候选指标和风险
- **THEN** 用户输入确认后系统才可以继续完整 workflow

#### Scenario: 用户拒绝理解结果
- **WHEN** 用户拒绝确认
- **THEN** 系统必须干净退出
- **THEN** 系统不得进入 Planner 或执行数据处理阶段
