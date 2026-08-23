## 背景与动机

多智能体自动化数据科学系统目前仅有架构脚手架和 Schema 合约，三个核心数据管道阶段——初步探索性数据分析（Preliminary EDA）、数据清洗（Data Cleaning）和特征工程（Feature Engineering）——均未实现（`NotImplementedError`），导致下游的模型构建和报告生成无法运行。这三个阶段处理 Titanic 数据集（`data/titanic/`），必须优先交付以解除整个流水线的阻塞。

## 变更内容

- 在 `app/agents/` 中实现 `run_preliminary_eda(state: GraphState) -> dict`，生成结构化 EDA 分析结果和工件
- 实现 `run_data_cleaning(state: GraphState) -> dict`，根据 EDA 发现对 Titanic 数据集进行缺失值填充、异常值处理和类型修正
- 实现 `run_feature_engineering(state: GraphState) -> dict`，为生存率分类任务构造并筛选预测特征
- 在 `app/prompts/` 中为上述阶段中涉及 LLM 的步骤添加提示词模板（如有需要）
- 在 `tests/unit/` 和 `tests/integration/` 中编写单元测试和集成冒烟测试
- 所有生成的工件必须遵循 `docs/engineering-specification.md` 中规定的保留路径命名

## 能力清单

### 新增能力

- `preliminary-eda`：分析 Titanic 训练/测试数据集，生成 Schema 摘要、缺失值报告、分布 EDA 发现，并将报告写入 `artifacts/eda/preliminary_eda.md`，图表写入 `artifacts/figures/`
- `data-cleaning`：根据 EDA 发现执行清洗变换（填充、类型转换、删除无关列），输出 `artifacts/data/cleaned_train.csv`、`artifacts/data/cleaned_test.csv` 和 `artifacts/data/cleaning_report.md`
- `feature-engineering`：构造预测特征（如称谓提取、家庭规模、票价分箱），筛选特征，输出 `artifacts/features/engineered_train.csv`、`artifacts/features/engineered_test.csv` 和 `artifacts/features/feature_summary.json`

### 修改能力

*（无——没有现有的规范级需求发生变更）*

## 影响范围

- **代码**：`app/agents/` 中新增三个阶段函数；可能需要在 `app/tools/` 中新增辅助工具（如 `exec_tools.py`、`file_tools.py`、`plot_tools.py`）
- **状态**：读写范围严格限定在 `data_state`、`feature_state` 和 `artifacts` 三个顶级键，遵守 `docs/io_contracts.md`
- **依赖**：`pandas`、`numpy`、`scikit-learn`（填充与编码）、`matplotlib`/`seaborn`（EDA 图表）
- **测试**：新增单元测试和集成冒烟测试，验证工件存在性和 State 键更新情况
- **无破坏性变更**：不新增顶级 `GraphState` 键，所有保留工件路径维持不变
