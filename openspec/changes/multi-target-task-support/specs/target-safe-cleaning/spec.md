## ADDED Requirements

### Requirement: 清洗工具自动保护目标列
`data_cleaning.py` 的工具执行循环 MUST 在执行任何 `drop_columns` 操作前，自动从参数 `columns` 列表中剔除 `state.target_spec.target_columns` 和 `state.target_spec.submission.id_columns` 中的列。

#### Scenario: LLM 建议删除目标列
- **WHEN** LLM Planner 生成的清洗计划中包含 `drop_columns(columns=["Exited", "Surname"])`
- **THEN** 实际执行时 MUST 仅删除 `["Surname"]`，`Exited` 被自动保护

#### Scenario: 多标签数据集的目标列保护
- **WHEN** `target_spec.target_columns` 为 `["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]`，且 LLM 建议删除其中任何一个
- **THEN** 所有 7 个目标列 MUST 被保留在训练集中

### Requirement: 训练集目标列完整性验证
`data_cleaning.py` 在清洗完成后 MUST 验证训练集仍包含所有 `target_spec.target_columns`。若任何目标列缺失，MUST 返回 `_error_envelope`。

#### Scenario: 清洗后目标列丢失
- **WHEN** 清洗操作意外导致目标列被删除（如通过缺失值相关的列联删除）
- **THEN** Agent MUST 返回错误，错误信息中包含缺失的目标列名

#### Scenario: 清洗后目标列完整
- **WHEN** 清洗正常完成
- **THEN** 训练集输出 CSV 中 MUST 包含所有 `target_spec.target_columns`

### Requirement: 测试集不得包含目标列
`data_cleaning.py` 在输出测试集时 MUST 确保测试集中不包含任何 `target_spec.target_columns`。若测试集原始数据中不包含目标列（正常情况），则无需额外操作。

#### Scenario: 正常测试集
- **WHEN** 测试集原始数据不包含目标列
- **THEN** 清洗后的测试集 MUST 不包含目标列

#### Scenario: 异常测试集（包含目标列）
- **WHEN** 测试集原始数据意外包含目标列
- **THEN** 清洗 Agent MUST 在输出前删除测试集中的目标列，并在 `data_quality_flags` 中记录警告

### Requirement: Data Cleaning 从 target_spec 读取目标信息
`run_data_cleaning()` MUST 从 `state.target_spec.target_columns` 获取目标列信息，不得从 `state.data_state.target_column` 或 `state.task_config.target_column` 读取。

#### Scenario: target_spec 已 resolved
- **WHEN** `state.target_spec.status == "resolved"` 且 `target_columns` 非空
- **THEN** 清洗 Agent 使用 `target_spec.target_columns` 作为受保护列表

#### Scenario: target_spec 未 resolved 时的降级
- **WHEN** `state.target_spec.status == "unresolved"`
- **THEN** Agent MUST 通过 `app/utils/target_scope.py` 的辅助函数尝试从旧字段降级获取目标列，并在 `data_quality_flags` 中记录警告

### Requirement: Planner Prompt 注入目标上下文
`build_data_cleaning_prompt()` MUST 接受 `target_spec_summary` 参数并注入到 Prompt 中，使 LLM 知道哪些列是目标列和 ID 列。

#### Scenario: 正常注入
- **WHEN** 调用 `build_data_cleaning_prompt(profile_str, target_spec_summary)`
- **THEN** 生成的 Prompt 中 MUST 包含目标列名、任务类型和 ID 列信息
