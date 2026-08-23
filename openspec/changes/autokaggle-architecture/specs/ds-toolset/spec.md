## 新增要求 (ADDED Requirements)

### Requirement: 稳定且确定性的宽表数据操作 (Deterministic Table Operations)
系统被要求纳入并内建所有关于核心数据集变换环节的底层 Python 工具库（如提取类 `fill_missing`，抹除类 `drop_column`、转换类 `encode_categorical`）。

#### Scenario: 中位数缺失插值处理
- **WHEN** 指定 `fill_missing` 功能工具且伴随使用 `median_by_group` 操作时
- **THEN** 底层 dataframe 直接安全有效地实现插补并妥善规避 Data Leakage 相关问题。

### Requirement: 图工作流内节点解析与挂载调用 (Node Executes Tools)
管道处理层的现有 LangGraph 节点需要无缝解释接收的这些工具字典，通过调度映射精准调用对应的工具子集方法函数。

#### Scenario: 将逻辑计划具象化为实际效果
- **WHEN** 此时 Data Cleaning 并收到了被传透的 工具指令链表 时
- **THEN** 其按原序列分发调度这些请求，从 DS 工具集中调用出那些真正在执行变更的引擎函数。
