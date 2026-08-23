## ADDED Requirements

### Requirement: 计算数据集 Schema 摘要
系统必须从 `data_state.current_train_path` 和 `data_state.current_test_path` 读取原始 CSV 文件，并将列名、数据类型和非空计数填充到 `data_state.schema_summary`。

#### Scenario: 有效数据时 Schema 摘要被正确填充
- **WHEN** `run_preliminary_eda` 以指向 Titanic 训练 CSV 的有效 `current_train_path` 被调用
- **THEN** `data_state.schema_summary` 必须包含所有列的条目，包括 `PassengerId`、`Survived`、`Pclass`、`Name`、`Sex`、`Age`、`SibSp`、`Parch`、`Ticket`、`Fare`、`Cabin` 和 `Embarked`

#### Scenario: 文件不存在时返回 Schema 错误包
- **WHEN** `current_train_path` 为空或指向不存在的文件
- **THEN** 节点必须返回 `error_type: "artifact_missing"` 的错误包，并且不得更新任何 State 字段

### Requirement: 计算缺失值摘要
系统必须计算训练集和测试集中每列的缺失值数量和比例，并将结果存储在 `data_state.schema_summary` 的 `missingness` 键下。

#### Scenario: 已知缺失列被正确检测
- **WHEN** Titanic 训练 CSV 被处理
- **THEN** `Age`、`Cabin` 和 `Embarked` 必须被报告为有缺失值，并附带各自的计数和比例

### Requirement: 生成 EDA 发现列表
系统必须向 `data_state.eda_findings` 追加结构化的 EDA 发现条目。每条条目必须是包含至少 `column`、`finding_type` 和 `detail` 键的字典。

#### Scenario: EDA 发现被正确填充
- **WHEN** `run_preliminary_eda` 无错误地完成
- **THEN** `data_state.eda_findings` 必须是非空列表，且对每个有缺失值或异常分布的列至少包含一条发现

### Requirement: 写入初步 EDA 报告工件
系统必须将 Markdown 摘要报告写入 `artifacts/eda/preliminary_eda.md`，并将 `artifacts["preliminary_eda_report"]` 更新为该路径。

#### Scenario: 报告工件被成功创建
- **WHEN** `run_preliminary_eda` 成功完成
- **THEN** 文件 `artifacts/eda/preliminary_eda.md` 必须存在，且 `artifacts["preliminary_eda_report"]` 必须等于 `"artifacts/eda/preliminary_eda.md"`

### Requirement: 写入 EDA 图表
系统必须生成至少一张分布图或缺失值热力图，保存至 `artifacts/figures/eda_<name>.png`，并将路径追加到 `artifacts["figures"]`。

#### Scenario: 图表路径被追加而非覆盖
- **WHEN** `run_preliminary_eda` 生成图表
- **THEN** `artifacts["figures"]` 必须以追加方式扩展，不得覆盖已有条目

### Requirement: 仅返回本节点所有的 State 更新
节点只能返回其所有的 State 键：`data_state`、`artifacts`、`current_phase` 和 `phase_history`。不得修改 `model_state`、`feature_state`、`review_state` 或 `report_state`。

#### Scenario: 无关 State 键保持不变
- **WHEN** `run_preliminary_eda` 返回其字典
- **THEN** 返回的字典不得包含 `model_state`、`feature_state`、`review_state` 或 `report_state` 作为键
