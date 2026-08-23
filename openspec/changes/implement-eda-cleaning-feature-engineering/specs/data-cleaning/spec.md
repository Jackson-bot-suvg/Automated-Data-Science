## ADDED Requirements

### Requirement: 填充缺失的 Age 值
系统必须使用按 `Pclass` 和 `Sex` 分组的中位数年龄填充缺失的 `Age` 值，并将填充策略记录在 `data_state.cleaning_summary` 中。

#### Scenario: 训练集的 Age 被正确填充
- **WHEN** `run_data_cleaning` 处理含缺失 `Age` 值的 Titanic 训练 CSV
- **THEN** 清洗后的训练数据集中 `Age` 列的缺失值数量必须为零
- **THEN** `data_state.cleaning_summary` 必须包含描述 `Age` 填充策略的条目

#### Scenario: 测试集的 Age 填充不存在数据泄露
- **WHEN** `run_data_cleaning` 处理 Titanic 测试 CSV
- **THEN** 必须使用从训练集拟合的分组中位数策略，不得从测试标签中泄露信息

### Requirement: 填充缺失的 Embarked 值
系统必须使用训练集的众数填充缺失的 `Embarked` 值，并将填充策略记录在 `data_state.cleaning_summary` 中。

#### Scenario: 训练集的 Embarked 被正确填充
- **WHEN** `run_data_cleaning` 处理 Titanic 训练 CSV
- **THEN** 清洗后的数据集中 `Embarked` 列的缺失值数量必须为零

### Requirement: 处理测试集中缺失的 Fare 值
系统必须使用训练集中对应 `Pclass` 的中位数票价填充测试集中缺失的 `Fare` 值。

#### Scenario: 测试集 Fare 被正确填充
- **WHEN** `run_data_cleaning` 处理 Titanic 测试 CSV 且某位乘客的 `Fare` 缺失
- **THEN** 该 `Fare` 必须使用该乘客 `Pclass` 对应的训练集中位数进行填充

### Requirement: 删除高缺失率的 Cabin 列
系统必须从训练集和测试集中删除 `Cabin` 列（其缺失率超过 77%），并将该删除操作记录在 `data_state.data_quality_flags` 中。

#### Scenario: 清洗后输出中不含 Cabin 列
- **WHEN** `run_data_cleaning` 完成
- **THEN** 清洗后的训练集和测试集 CSV 均不得包含 `Cabin` 列
- **THEN** `data_state.data_quality_flags` 必须包含一条记录 `Cabin` 被删除的标志条目

### Requirement: 删除无关标识符列
系统必须从训练数据集中删除 `PassengerId`、`Name` 和 `Ticket`（这些是无直接预测信号的标识符）。测试集中的 `PassengerId` 必须保留以便提交对齐。

#### Scenario: 清洗后训练集中不含 Name 和 Ticket
- **WHEN** `run_data_cleaning` 生成 `cleaned_train.csv`
- **THEN** `Name` 和 `Ticket` 不得作为列存在

#### Scenario: 测试集中保留 PassengerId
- **WHEN** `run_data_cleaning` 生成 `cleaned_test.csv`
- **THEN** `PassengerId` 必须作为列保留在测试输出中

### Requirement: 对类别型列进行编码
系统必须将 `Sex` 编码为二元整数（male=0，female=1），将 `Embarked` 进行独热编码或序号编码。编码决策必须记录在 `data_state.cleaning_summary` 中。

#### Scenario: Sex 列仅包含 0 或 1
- **WHEN** `run_data_cleaning` 完成
- **THEN** 清洗后数据集中的 `Sex` 列只能包含值 `0` 和 `1`

### Requirement: 写入清洗工件文件
系统必须将清洗后的数据集写入 `artifacts/data/cleaned_train.csv` 和 `artifacts/data/cleaned_test.csv`，并将清洗报告写入 `artifacts/data/cleaning_report.md`。三个工件 State 键必须全部更新。

#### Scenario: 全部三个清洗工件被成功创建
- **WHEN** `run_data_cleaning` 成功完成
- **THEN** `artifacts/data/cleaned_train.csv`、`artifacts/data/cleaned_test.csv` 和 `artifacts/data/cleaning_report.md` 必须全部存在
- **THEN** `artifacts["cleaned_train_data"]`、`artifacts["cleaned_test_data"]` 和 `artifacts["cleaning_report"]` 必须各自持有对应路径

### Requirement: 更新 data_state 路径指针
系统必须将 `data_state.current_train_path` 和 `data_state.current_test_path` 更新为指向新写入的清洗文件。

#### Scenario: 清洗后路径指针被更新
- **WHEN** `run_data_cleaning` 返回其 State 字典
- **THEN** `data_state.current_train_path` 必须等于 `"artifacts/data/cleaned_train.csv"`
- **THEN** `data_state.current_test_path` 必须等于 `"artifacts/data/cleaned_test.csv"`

### Requirement: 仅返回本节点所有的 State 更新
节点只能返回 `docs/io_contracts.md` 规定的所有键，不得修改 `feature_state`、`model_state`、`review_state` 或 `report_state`。

#### Scenario: 无关 State 键保持不变
- **WHEN** `run_data_cleaning` 返回其字典
- **THEN** 返回的字典不得包含 `feature_state`、`model_state`、`review_state` 或 `report_state` 作为键
