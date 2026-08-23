## ADDED Requirements

### Requirement: 从 Name 列中提取乘客称谓
系统必须在 `Name` 列被数据清洗阶段删除之前，通过提取称谓词（如 Mr、Mrs、Miss、Master）来派生 `Title` 特征。低频称谓必须合并为 `Other` 类别，`Title` 列必须进行序号编码。

#### Scenario: 工程输出中包含 Title 特征
- **WHEN** `run_feature_engineering` 完成
- **THEN** 工程化的训练集和测试集必须包含整数编码的 `Title` 列
- **THEN** 工程化输出中不得包含 `Name` 列

### Requirement: 构造 FamilySize 特征
系统必须创建 `FamilySize` 特征，计算方式为 `SibSp + Parch + 1`（含乘客本人）。同时必须创建二元特征 `IsAlone`，当 `FamilySize == 1` 时为 `1`，否则为 `0`。

#### Scenario: FamilySize 被正确计算
- **WHEN** 某位乘客的 `SibSp=1`，`Parch=2`
- **THEN** `FamilySize` 必须等于 `4`，`IsAlone` 必须等于 `0`

#### Scenario: IsAlone 被正确计算
- **WHEN** 某位乘客的 `SibSp=0`，`Parch=0`
- **THEN** `FamilySize` 必须等于 `1`，`IsAlone` 必须等于 `1`

### Requirement: 将 Fare 分箱为四分位段
系统必须通过将 `Fare` 分为 4 个基于分位数的区间来创建 `FareBand` 特征。区间边界必须在训练集上拟合并同等应用于测试集以防止数据泄露。原始 `Fare` 列在分箱后必须被删除。

#### Scenario: FareBand 替代工程输出中的 Fare
- **WHEN** `run_feature_engineering` 完成
- **THEN** 工程化数据集必须包含整数标签为 `0`–`3` 的 `FareBand` 列
- **THEN** 工程化输出中不得包含 `Fare` 列

#### Scenario: 区间边界仅在训练集上拟合
- **WHEN** `FareBand` 被计算用于测试数据集
- **THEN** 区间边界必须完全来自训练集分布

### Requirement: 将 Age 分箱为年龄段
系统必须通过将 `Age` 分为 5 个等宽或基于分位数的区间来创建 `AgeBand` 特征。区间边界必须在填充后的训练集上拟合。原始 `Age` 列必须被删除。

#### Scenario: AgeBand 替代工程输出中的 Age
- **WHEN** `run_feature_engineering` 完成
- **THEN** 工程化数据集必须包含 `AgeBand` 列
- **THEN** 工程化输出中不得包含 `Age` 列

### Requirement: 创建 FamilySize 后删除 SibSp 和 Parch
创建 `FamilySize` 和 `IsAlone` 后，系统必须删除 `SibSp` 和 `Parch` 以降低共线性。

#### Scenario: 工程输出中不含 SibSp 和 Parch
- **WHEN** `run_feature_engineering` 完成
- **THEN** 工程化训练集和测试集中均不得包含 `SibSp` 或 `Parch` 列

### Requirement: 记录已选择和已删除的特征
系统必须用最终特征列名填充 `feature_state.selected_features`，用特征工程阶段删除的列名填充 `feature_state.dropped_features`。

#### Scenario: selected_features 被正确填充
- **WHEN** `run_feature_engineering` 完成
- **THEN** `feature_state.selected_features` 必须是非空列表，包含工程化训练数据集中所有列（不含目标列 `Survived`）

#### Scenario: dropped_features 被正确填充
- **WHEN** `run_feature_engineering` 完成
- **THEN** `feature_state.dropped_features` 至少必须包含 `SibSp`、`Parch`、`Age` 和 `Fare`

### Requirement: 写入工程化工件文件和特征摘要
系统必须将工程化训练集和测试集写入 `artifacts/features/engineered_train.csv` 和 `artifacts/features/engineered_test.csv`。必须将记录所有变换的 JSON 特征摘要写入 `artifacts/features/feature_summary.json`。三个工件 State 键必须全部更新。

#### Scenario: 全部特征工件被成功创建
- **WHEN** `run_feature_engineering` 成功完成
- **THEN** `artifacts/features/engineered_train.csv`、`artifacts/features/engineered_test.csv` 和 `artifacts/features/feature_summary.json` 必须全部存在
- **THEN** `artifacts["feature_summary"]` 必须等于 `"artifacts/features/feature_summary.json"`

### Requirement: 更新 feature_state 路径指针
系统必须将 `feature_state.engineered_train_path` 和 `feature_state.engineered_test_path` 更新为工程化文件的路径。

#### Scenario: 路径指针被更新
- **WHEN** `run_feature_engineering` 返回其字典
- **THEN** `feature_state.engineered_train_path` 必须等于 `"artifacts/features/engineered_train.csv"`
- **THEN** `feature_state.engineered_test_path` 必须等于 `"artifacts/features/engineered_test.csv"`

### Requirement: 不得从测试集泄露数据
所有拟合操作（区间边界、编码器、分组统计）必须仅在训练数据集上计算，然后应用于测试数据集。

#### Scenario: 测试集变换使用训练集拟合的参数
- **WHEN** 任何分箱或编码被应用于测试集
- **THEN** 参数（区间边界、类别映射）必须与从训练集推导出的完全一致

### Requirement: 仅返回本节点所有的 State 更新
节点只能返回 `docs/io_contracts.md` 规定的所有键，不得修改 `data_state`、`model_state`、`review_state` 或 `report_state`。

#### Scenario: 无关 State 键保持不变
- **WHEN** `run_feature_engineering` 返回其字典
- **THEN** 返回的字典不得包含 `data_state`、`model_state`、`review_state` 或 `report_state` 作为键
