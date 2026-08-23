## ADDED Requirements

### Requirement: selected_features 排除目标列和 ID 列
`run_feature_engineering()` 在生成 `selected_features` 时 MUST 排除 `state.target_spec.target_columns` 和 `state.target_spec.submission.id_columns` 中的所有列。

#### Scenario: 单目标数据集
- **WHEN** `target_spec.target_columns` 为 `["Exited"]`，`id_columns` 为 `["id", "CustomerId"]`
- **THEN** `selected_features` 中 MUST NOT 包含 `Exited`、`id`、`CustomerId`

#### Scenario: 多标签数据集
- **WHEN** `target_spec.target_columns` 包含 7 个目标列
- **THEN** `selected_features` 中 MUST NOT 包含任何目标列

### Requirement: 编码工具不得编码目标列
特征工程工具执行循环 MUST 在执行 `encode_categorical` 操作前，检查目标列是否在参数 `column` 中。若目标列被传入编码工具，MUST 跳过该步骤。

#### Scenario: LLM 建议编码目标列
- **WHEN** LLM Planner 生成的特征计划中包含 `encode_categorical(column="NObeyesdad", method="label")`，且 `NObeyesdad` 是目标列
- **THEN** 该步骤 MUST 被跳过，并在 `data_quality_flags` 中记录跳过原因

#### Scenario: 正常编码非目标列
- **WHEN** 编码目标为 `Geography`（非目标列）
- **THEN** 编码正常执行

### Requirement: 训练集/测试集特征 Schema 对齐验证
`run_feature_engineering()` 在输出前 MUST 验证训练集和测试集的特征列（排除目标列后）完全一致。

#### Scenario: Schema 一致
- **WHEN** 训练集特征列 `["A", "B", "C"]` 与测试集特征列 `["A", "B", "C"]` 一致
- **THEN** 正常输出

#### Scenario: Schema 不一致
- **WHEN** 训练集有 `["A", "B", "C"]` 而测试集有 `["A", "B"]`（缺少 `C`）
- **THEN** Agent MUST 在 `data_quality_flags` 中记录不一致的列名，并尝试对齐（为测试集补充缺失列填零或删除多余列）

### Requirement: Feature Engineering 从 target_spec 读取目标信息
`run_feature_engineering()` MUST 从 `state.target_spec.target_columns` 获取目标列信息。

#### Scenario: target_spec 已 resolved
- **WHEN** `state.target_spec.status == "resolved"`
- **THEN** 使用 `target_spec.target_columns` 和 `target_spec.submission.id_columns` 进行列排除

#### Scenario: target_spec 未 resolved 时的降级
- **WHEN** `state.target_spec.status == "unresolved"`
- **THEN** 通过 `app/utils/target_scope.py` 的辅助函数从旧字段降级获取

### Requirement: 回归场景的特征转换兼容
特征工程工具 MUST 在回归场景中正确处理连续目标变量，不得对目标变量执行分类相关操作（如 `group_rare_categories`）。

#### Scenario: house_prices 回归数据集
- **WHEN** `task_family` 为 `regression`，目标列 `SalePrice` 为连续值
- **THEN** 工具执行循环中 MUST NOT 对 `SalePrice` 执行 `group_rare_categories` 或 `encode_categorical`

### Requirement: Planner Prompt 注入目标上下文
`build_feature_engineering_prompt()` MUST 接受 `target_spec_summary` 参数并注入到 Prompt 中，使 LLM 知道哪些列是目标列、ID 列以及当前任务类型。

#### Scenario: 正常注入
- **WHEN** 调用 `build_feature_engineering_prompt(profile_str, target_spec_summary)`
- **THEN** 生成的 Prompt 中 MUST 包含目标列名、ID 列名和任务类型信息
