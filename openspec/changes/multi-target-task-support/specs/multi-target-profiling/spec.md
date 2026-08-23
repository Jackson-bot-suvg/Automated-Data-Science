## ADDED Requirements

### Requirement: 多目标列画像支持
`profile_dataset()` 函数 SHALL 接受 `target_cols: list[str] | None` 参数，支持零个、一个或多个目标列的数据画像生成。

#### Scenario: 单目标分类数据集
- **WHEN** `target_cols` 为 `["Exited"]` 且该列为二值列
- **THEN** 画像 JSON 中 MUST 包含 `class_distribution` 字段，内容为该目标列的类别占比字典

#### Scenario: 多目标（多标签）数据集
- **WHEN** `target_cols` 为 `["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]`
- **THEN** 画像 JSON 中 MUST 包含每个目标列各自的 `class_distribution`，且 MUST 包含 `label_cooccurrence` 字段（标签间皮尔逊相关矩阵的上三角）

#### Scenario: 回归数据集
- **WHEN** `target_cols` 为 `["SalePrice"]` 且该列为连续数值型
- **THEN** 画像 JSON 中 MUST 包含 `target_statistics` 字段，内容包括均值、中位数、标准差、偏度和峰度；MUST NOT 包含 `class_distribution`

#### Scenario: 未指定目标列
- **WHEN** `target_cols` 为 `None` 或空列表
- **THEN** 画像 JSON 中 `target_variable` MUST 为 `None`，不生成任何目标相关统计

### Requirement: 向后兼容旧签名
`profile_dataset()` MUST 保留 `target_col: str` 参数作为 deprecated alias。当调用方仍使用旧签名时，函数 SHALL 自动将其转换为 `target_cols=[target_col]`。

#### Scenario: 旧签名调用
- **WHEN** 调用方使用 `profile_dataset(df, target_col="Survived")`
- **THEN** 函数 MUST 正常工作并返回与 `target_cols=["Survived"]` 相同的结果

### Requirement: Preliminary EDA 从 target_spec 读取目标列
`run_preliminary_eda()` MUST 从 `state.target_spec.target_columns` 获取目标列，不得硬编码任何列名。

#### Scenario: bank_churn 数据集
- **WHEN** `state.target_spec.target_columns` 为 `["Exited"]`
- **THEN** EDA 画像中 `target_variable` 为 `["Exited"]`，且生成该列的类别分布

#### Scenario: plate_defect 多标签数据集
- **WHEN** `state.target_spec.target_columns` 为 7 个目标列
- **THEN** EDA 画像中 MUST 包含每个目标列的分布和标签共现矩阵

#### Scenario: house_prices 回归数据集
- **WHEN** `state.target_spec.target_columns` 为 `["SalePrice"]` 且 `task_family` 为 `regression`
- **THEN** EDA 画像中 MUST 包含连续值统计而非类别分布

### Requirement: 任务类型感知的画像字段命名
Profiler MUST 根据目标列的数据类型自动判断输出字段：数值连续型目标使用 `target_statistics`，类别/二值型目标使用 `class_distribution`。

#### Scenario: 自动判断分类目标
- **WHEN** 目标列 `nunique <= 20` 或为 object/bool 类型
- **THEN** 生成 `class_distribution`

#### Scenario: 自动判断回归目标
- **WHEN** 目标列为 float/int 且 `nunique > 20`
- **THEN** 生成 `target_statistics`（均值、中位数、标准差、偏度、峰度）
