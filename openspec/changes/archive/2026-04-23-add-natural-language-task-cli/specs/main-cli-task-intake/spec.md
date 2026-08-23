## ADDED Requirements

### Requirement: CLI 接受自然语言任务和数据集目录
系统必须提供 `main.py` 作为主入口，允许用户通过 CLI 输入自然语言任务和数据集位置，并启动完整 workflow。

#### Scenario: 用户以自然语言任务和目录启动
- **WHEN** 用户运行 `python main.py --task "用这个数据集预测客户流失" --dataset data/bank_churn`
- **THEN** 系统必须读取 CLI 参数
- **THEN** 系统必须构造初始 workflow state
- **THEN** 系统必须先执行 Reader 和 Background Reviewer 理解/审核阶段
- **THEN** 用户确认后系统必须继续运行完整 workflow 到 business report

### Requirement: TaskIntake 保留原始任务和数据集根目录
系统必须有独立 TaskIntake 层负责把 CLI 输入转换为初始状态，并保留用户原始自然语言任务和数据集根目录。

#### Scenario: 数据集参数是目录
- **WHEN** `--dataset` 指向一个目录
- **THEN** TaskIntake 必须在现有顶层 `task_config` 内保存该目录作为 dataset root，或使用等价的现有字段承载该语义
- **THEN** TaskIntake 不得新增 `GraphState` 顶层字段
- **THEN** TaskIntake 不得把目录路径伪装成 `dataset_paths.train`
- **THEN** Reader 必须负责读取目录并判断实际数据文件

#### Scenario: 用户提供目标提示
- **WHEN** 用户传入 `--target` 且内容可能是自然语言
- **THEN** TaskIntake 必须保留该目标提示
- **THEN** Reader 必须结合目标提示和真实列名判断具体 target column

### Requirement: Reader 根据数据集目录进行受控 ReAct 理解
Reader 必须能根据 dataset root 读取目录、识别数据文件、读取有界元数据，并生成任务理解。

#### Scenario: 标准 Kaggle 风格目录
- **WHEN** 数据集目录包含 `train.csv`、`test.csv` 和 `sample_submission.csv`
- **THEN** Reader 必须识别训练集、测试集和提交样例路径
- **THEN** Reader 必须填充 `data_state.raw_train_path`、`data_state.raw_test_path` 和相关 `dataset_paths`
- **THEN** Reader 必须基于有界 CSV 元数据推断 target、task type 和 candidate metrics

#### Scenario: 目录中存在说明文件
- **WHEN** 数据集目录包含 `overview.txt` 或 `data_description.txt`
- **THEN** Reader 可以读取有限长度的说明文本作为任务理解上下文
- **THEN** Reader 不得把完整大文件直接放入 LLM prompt

### Requirement: 缺少 test.csv 时以 warning 继续
系统必须允许第一版在只有 `train.csv` 的数据集目录上继续运行，但必须明确记录风险和限制。

#### Scenario: 数据集目录没有测试集
- **WHEN** 数据集目录包含可读 `train.csv` 但没有 `test.csv`
- **THEN** Reader 必须记录缺少外部测试集的 warning/risk
- **THEN** Background Reviewer 不得仅因缺少 `test.csv` 返回 fatal
- **THEN** Reviewer 和 artifact 验收不得强制要求真正的外部测试集 submission 文件
- **THEN** 后续报告必须说明评估依赖训练集内部验证或交叉验证，不能代表独立外部测试表现

### Requirement: target 不确定时允许用户自然语言补充
系统必须在 target 无法可靠推断时，通过 CLI 直接让用户补充自然语言目标，并重新执行 Reader 理解和 Background Reviewer 审核。

#### Scenario: Reader 无法确定目标列
- **WHEN** Reader 无法从 train/test 列差异、任务文本或目标提示中确定 target column
- **THEN** Background Reviewer 必须返回 retry
- **THEN** CLI 必须提示用户用自然语言说明想预测的目标
- **THEN** 系统必须将用户补充内容传回 Reader 重新理解

#### Scenario: 自然语言补充仍无法确定
- **WHEN** 用户补充自然语言目标后 Reader 仍无法映射到具体列名
- **THEN** CLI 必须展示候选列
- **THEN** 用户必须能输入精确列名以继续

### Requirement: 进入 Planner 前必须有人类确认
系统必须在 Background Reviewer 审核通过后，将 Reader 理解结果展示给用户确认，确认后才进入 Planner。

#### Scenario: 用户确认理解结果
- **WHEN** Background Reviewer 返回 approved
- **THEN** CLI 必须展示任务摘要、任务类型、目标列、数据文件、候选指标和风险
- **THEN** 用户输入确认后系统才可以继续完整 workflow

#### Scenario: 用户拒绝理解结果
- **WHEN** 用户拒绝确认
- **THEN** 系统必须干净退出
- **THEN** 系统不得进入 Planner 或执行数据处理阶段

### Requirement: CLI 默认生成 business report
主入口必须默认运行完整 workflow，并在结束时输出 business report 路径。

#### Scenario: CLI 调用 graph 时携带 thread id
- **WHEN** CLI 准备启动 LangGraph workflow
- **THEN** 系统必须设置 `runtime_state.thread_id`
- **THEN** graph 调用必须传入 `configurable.thread_id`
- **THEN** 该 thread id 必须能用于后续持久化、人类确认恢复或运行追踪

#### Scenario: 完整 workflow 成功
- **WHEN** 用户确认 Reader 理解并且 workflow 成功完成
- **THEN** 系统必须生成 technical report
- **THEN** 系统必须生成 business report
- **THEN** CLI 必须打印 business report、technical report 和 prediction/submission 相关路径

### Requirement: 维护 docs 与 skills 文档
本次变更必须同步维护 `docs/` 和 `docs/skills/` 下与新入口、新职责边界和开发流程相关的文档。

#### Scenario: 文档反映新的主入口
- **WHEN** `main.py` 和 TaskIntake 行为完成
- **THEN** `README.md` 和 `README_cn.md` 必须包含新的 `python main.py --task ... --dataset ...` 用法
- **THEN** `docs/task_intake_layer.md` 必须说明 TaskIntake、Reader 目录理解、Background Reviewer 审核和人类确认流程
- **THEN** `docs/graph_design.md` 必须说明 CLI 默认运行到 business report

#### Scenario: skills 文档保持可维护
- **WHEN** 新入口改变模块职责或开发流程
- **THEN** `docs/skills/` 中相关 skill 文档必须检查并更新
- **THEN** 文档必须避免继续暗示只有 Titanic 硬编码入口或仅运行到 technical report
