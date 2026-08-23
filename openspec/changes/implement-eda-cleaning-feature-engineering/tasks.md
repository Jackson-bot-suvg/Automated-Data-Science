## 1. 准备工作与目录结构

- [x] 1.1 在项目根目录创建工件子目录：`artifacts/eda/`、`artifacts/data/`、`artifacts/features/`、`artifacts/figures/`
- [x] 1.2 创建 `app/agents/preliminary_eda.py` 文件，包含模块文档字符串和 `run_preliminary_eda` 函数骨架
- [x] 1.3 创建 `app/agents/data_cleaning.py` 文件，包含模块文档字符串和 `run_data_cleaning` 函数骨架
- [x] 1.4 创建 `app/agents/feature_engineering.py` 文件，包含模块文档字符串和 `run_feature_engineering` 函数骨架
- [x] 1.5 在 `app/agents/__init__.py` 中导出三个新函数

## 2. 实现 Preliminary EDA 节点

- [x] 2.1 从 `state.data_state.current_train_path` 和 `current_test_path` 读取 CSV 文件；若路径为空或文件不存在则返回标准错误包（`error_type: "artifact_missing"`）
- [x] 2.2 计算训练集和测试集各列的数据类型、非空计数，填充 `data_state.schema_summary`
- [x] 2.3 计算每列的缺失值数量和比例，存入 `data_state.schema_summary["missingness"]`
- [x] 2.4 生成结构化 EDA 发现列表，追加到 `data_state.eda_findings`（每条含 `column`、`finding_type`、`detail` 字段）
- [x] 2.5 使用 `os.makedirs(..., exist_ok=True)` 确保 `artifacts/eda/` 和 `artifacts/figures/` 目录存在
- [x] 2.6 用 seaborn/matplotlib 生成缺失值热力图，保存至 `artifacts/figures/eda_missingness.png`，将路径追加到 `artifacts["figures"]`
- [x] 2.7 用 seaborn/matplotlib 生成 `Survived` 与 `Pclass`、`Sex`、`Age`、`Fare` 的分布图，各自保存至 `artifacts/figures/eda_<name>.png`，路径全部追加到 `artifacts["figures"]`
- [x] 2.8 将 EDA 摘要报告写入 `artifacts/eda/preliminary_eda.md`，更新 `artifacts["preliminary_eda_report"]`
- [x] 2.9 更新 `current_phase` 为 `"preliminary_eda"`，向 `phase_history` 追加该阶段名称
- [x] 2.10 验证返回字典只包含 `data_state`、`artifacts`、`current_phase`、`phase_history` 四个顶级键

## 3. 实现 Data Cleaning 节点

- [x] 3.1 从 `state.data_state.current_train_path` 读取训练 CSV；从 `current_test_path` 读取测试 CSV
- [x] 3.2 按 `Pclass` 和 `Sex` 分组计算训练集 `Age` 中位数，以字典形式保存拟合值；对训练集和测试集填充缺失 `Age`，将策略记录到 `cleaning_summary`
- [x] 3.3 计算训练集 `Embarked` 众数，填充训练集和测试集缺失 `Embarked`，记录到 `cleaning_summary`
- [x] 3.4 计算各 `Pclass` 的训练集 `Fare` 中位数，填充测试集缺失 `Fare`，记录到 `cleaning_summary`
- [x] 3.5 从训练集和测试集中删除 `Cabin` 列，向 `data_state.data_quality_flags` 追加删除标志
- [x] 3.6 从训练集删除 `PassengerId` 和 `Ticket`；**保留 `Name` 列**（由特征工程阶段处理），在 `cleaning_report.md` 中注明"Name 列延迟删除"
- [x] 3.7 将 `Sex` 编码为二元整数（male=0，female=1）；将 `Embarked` 进行序号编码；记录编码映射到 `cleaning_summary`
- [x] 3.8 用 `os.makedirs(..., exist_ok=True)` 确保 `artifacts/data/` 目录存在
- [x] 3.9 将清洗后训练集写入 `artifacts/data/cleaned_train.csv`，更新 `artifacts["cleaned_train_data"]`
- [x] 3.10 将清洗后测试集写入 `artifacts/data/cleaned_test.csv`，更新 `artifacts["cleaned_test_data"]`
- [x] 3.11 将清洗报告（包含所有 `cleaning_summary` 内容）写入 `artifacts/data/cleaning_report.md`，更新 `artifacts["cleaning_report"]`
- [x] 3.12 将 `data_state.current_train_path` 和 `current_test_path` 更新为新清洗文件路径
- [x] 3.13 更新 `current_phase` 为 `"data_cleaning"`，向 `phase_history` 追加该阶段名称
- [x] 3.14 验证返回字典只包含 `data_state`、`artifacts`、`current_phase`、`phase_history` 四个顶级键

## 4. 实现 Feature Engineering 节点

- [x] 4.1 从 `state.data_state.current_train_path` 和 `current_test_path`（即清洗后的文件）读取 CSV
- [x] 4.2 从 `Name` 列提取称谓词（Mr/Mrs/Miss/Master），将低频称谓合并为 `Other`，进行序号编码生成 `Title` 列；从训练集和测试集删除 `Name` 列
- [x] 4.3 计算 `FamilySize = SibSp + Parch + 1`，计算 `IsAlone = (FamilySize == 1).astype(int)`
- [x] 4.4 删除 `SibSp` 和 `Parch` 列
- [x] 4.5 使用 `pd.qcut(..., retbins=True)` 在训练集上拟合 `FareBand` 分位数边界（4 箱），用 `pd.cut` 将同一边界应用于测试集；删除原始 `Fare` 列；将边界值记录到 `feature_transform_summary`
- [x] 4.6 使用 `pd.qcut(..., retbins=True)` 在训练集上拟合 `AgeBand` 分位数边界（5 箱），用 `pd.cut` 将同一边界应用于测试集；删除原始 `Age` 列；将边界值记录到 `feature_transform_summary`
- [x] 4.7 用 `os.makedirs(..., exist_ok=True)` 确保 `artifacts/features/` 目录存在
- [x] 4.8 将工程化训练集写入 `artifacts/features/engineered_train.csv`，更新 `feature_state.engineered_train_path` 和 `artifacts["feature_summary"]`（路径）
- [x] 4.9 将工程化测试集写入 `artifacts/features/engineered_test.csv`，更新 `feature_state.engineered_test_path`
- [x] 4.10 生成 `feature_summary.json`（含所有变换描述和 `feature_transform_summary`），写入 `artifacts/features/feature_summary.json`
- [x] 4.11 填充 `feature_state.selected_features`（训练集最终列名，不含 `Survived`）和 `feature_state.dropped_features`（含 `SibSp`、`Parch`、`Age`、`Fare`、`Name` 等）
- [x] 4.12 更新 `current_phase` 为 `"feature_engineering"`，向 `phase_history` 追加该阶段名称
- [x] 4.13 验证返回字典只包含 `feature_state`、`artifacts`、`current_phase`、`phase_history` 四个顶级键

## 5. 单元测试

- [x] 5.1 创建 `tests/unit/test_preliminary_eda.py`：验证 `run_preliminary_eda` 在有效 State 下返回包含 `data_state` 和 `artifacts` 键的字典；验证文件不存在时返回错误包
- [x] 5.2 创建 `tests/unit/test_data_cleaning.py`：验证 `run_data_cleaning` 在有效 State 下 `Age` 列无缺失值；验证 `Cabin` 列不存在；验证 `Sex` 列只含 `0` 和 `1`；验证路径指针已更新
- [x] 5.3 创建 `tests/unit/test_feature_engineering.py`：验证 `run_feature_engineering` 在有效 State 下输出含 `Title`、`FamilySize`、`IsAlone`、`FareBand`、`AgeBand`；验证 `SibSp`、`Parch`、`Age`、`Fare`、`Name` 列已被删除；验证 `feature_state.selected_features` 非空
- [x] 5.4 在每个单元测试文件中，验证返回字典不含无关顶级 State 键（`model_state`、`review_state` 等）

## 6. 集成测试

- [x] 6.1 更新 `tests/integration/test_state_smoke.py`：验证三个新节点的返回值可与 `GraphState` 合并而不报模式错误
- [x] 6.2 创建 `tests/integration/test_pipeline_eda_to_features.py`：以真实 Titanic CSV 初始化 State，顺序调用三个节点，验证全部工件文件存在（`preliminary_eda.md`、`cleaned_train.csv`、`cleaned_test.csv`、`cleaning_report.md`、`engineered_train.csv`、`engineered_test.csv`、`feature_summary.json`）
- [x] 6.3 在集成测试中验证 `phase_history` 包含三个阶段名称且无重复（验证 Reducer 追加正确）
- [x] 6.4 在集成测试中验证测试集 `engineered_test.csv` 的列集合与训练集一致（除去 `Survived`）
