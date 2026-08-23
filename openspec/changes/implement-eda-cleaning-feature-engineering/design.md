## 背景

当前仓库是一个基于 LangGraph 的多智能体自动化数据科学系统，目前仅有架构脚手架和 Pydantic Schema 合约。实际业务节点（Agent Nodes）均为占位符（`NotImplementedError`）。

本次变更需要在现有脚手架之上，为 Titanic 生存率分类任务实现三个核心数据管道阶段：
- **Preliminary EDA**：数据集摘要分析与发现记录
- **Data Cleaning**：缺失值处理、列删除与类别编码
- **Feature Engineering**：特征构造、分箱与特征选择

所有实现必须严格遵守 `docs/engineering-specification.md` 和 `docs/io_contracts.md` 的约束：通过 `GraphState` 共享状态、使用保留工件路径、返回局部 State 更新。

---

## Goals / Non-Goals

**目标：**
- 实现三个节点函数：`run_preliminary_eda`、`run_data_cleaning`、`run_feature_engineering`
- 每个节点读取规范规定的 State 输入字段，执行逻辑，并写入规范规定的 State 输出字段
- 所有生成的文件使用 `docs/engineering-specification.md` 第 9 条规定的保留工件路径
- 为每个节点提供至少一个单元测试和一个集成冒烟测试
- 确保训练集拟合参数与测试集变换之间无数据泄露

**非目标：**
- 不实现 LLM 驱动的 EDA 或自动化特征发现（本次使用规则驱动的确定性逻辑）
- 不实现模型构建、Reviewer 或报告生成节点（后续 Change 处理）
- 不修改 `GraphState` 的顶级键结构
- 不实现 Web UI 或交互式可视化

---

## Decisions

### 决策 1：三个函数独立文件 vs. 合并入 `developer.py`

**选择**：为每个阶段创建独立的模块文件：
- `app/agents/preliminary_eda.py`
- `app/agents/data_cleaning.py`
- `app/agents/feature_engineering.py`

**理由**：`developer.py` 是一个通用占位符，不代表任何明确的业务职责。将三个阶段合并进一个文件会违反单一职责原则（SRP），并使测试边界模糊。独立文件符合 `docs/architecture.md` 的"按职责隔离"原则，也便于后续团队成员并行开发不同阶段。

**备选方案**：全部写在 `developer.py` 中 → 否决，职责不清晰，文件过长。

---

### 决策 2：数据清洗参数的管理方式

**选择**：使用 Python 字典在 `run_data_cleaning` 函数内存储拟合的参数（如分组中位数年龄、众数 Embarked），并将这些参数在返回的 State 字典的 `data_state.cleaning_summary` 中以 JSON 可序列化格式记录。

**理由**：LangGraph 的检查点（Checkpointer）要求 State 中的所有内容必须可序列化。不能将 scikit-learn 的 Imputer 对象直接存入 `GraphState`（违反 Hard Constraint §13）。因此，拟合值以字典形式保存，并在特征工程阶段从 `cleaning_summary` 读取复用。

**备选方案**：将 sklearn 对象 pickle 后存入 State → 否决，会引入不可序列化对象，违反工程规范。

---

### 决策 3：特征工程中 Title 提取的时机

**选择**：`Title` 提取发生在 `run_feature_engineering` 中，读取 `data_state.current_train_path`（即清洗后的文件）之前，`Name` 列被清洗阶段物理删除之前。

**实现方式**：数据清洗阶段**不删除 `Name` 列**，仅在清洗报告中标记为"待特征工程处理"。特征工程阶段读取当前文件时仍可访问 `Name`，完成 `Title` 提取后再删除。

**理由**：避免跨阶段隐式依赖（清洗阶段不应知道特征工程需要 `Name`），职责边界清晰。

**备选方案**：在清洗阶段直接提取 `Title` → 否决，职责越界，清洗节点不该生产特征。

---

### 决策 4：EDA 图表库选择

**选择**：使用 `matplotlib` + `seaborn` 生成图表，保存为 PNG 文件至 `artifacts/figures/`。

**理由**：无需运行环境支持交互，生成的图表文件可直接被后续报告生成阶段引用，且路径可追加写入 `artifacts["figures"]`（append-only Reducer 安全）。

**备选方案**：使用 `plotly` 生成交互图 → 否决，HTML 文件不适合 artifact 路径约束且无法直接嵌入 Markdown 报告。

---

### 决策 5：分箱参数传递方式（防数据泄露）

**选择**：在 `run_feature_engineering` 内部，先用训练集计算 `FareBand` 和 `AgeBand` 的分位数边界（`pd.qcut` 的 `retbins=True`），将边界值存入本地变量，再用 `pd.cut` 将同一边界应用于测试集。边界值记录进 `feature_state.feature_transform_summary`（JSON 可序列化）。

**理由**：严格防止测试集信息影响分箱边界，符合 spec 中"不得从测试集泄露数据"的要求。

---

## Risks / Trade-offs

| 风险 | 缓解措施 |
|---|---|
| `Name` 列在清洗阶段未被删除，其他开发者误以为清洗不完整 | 在 `cleaning_report.md` 中明确注明 `Name` 列由特征工程阶段处理并删除 |
| `cleaning_summary` 中存储的分组中位数与测试集实际分布差异过大，导致填充效果差 | 属于模型质量问题，不是架构问题；在清洗报告中记录填充值以便 Reviewer 审查 |
| 低频 Title 分组策略（`Other`）可能影响模型效果 | 按 spec 实现确定性规则，调参留给模型构建阶段 |
| `artifacts/figures/` 目录可能在首次运行时不存在 | 节点在写图前用 `os.makedirs(..., exist_ok=True)` 创建目录 |
| Titanic 测试集无 `Survived` 列，集成测试无法验证端到端准确率 | 集成测试仅验证工件存在性和 State 键更新，不验证模型性能 |

---

## Migration Plan

本次变更为纯新增实现，不修改任何现有接口：

1. 创建三个新 Agent 模块文件（`preliminary_eda.py`、`data_cleaning.py`、`feature_engineering.py`）
2. 创建对应的工件输出目录（`artifacts/eda/`、`artifacts/data/`、`artifacts/features/`、`artifacts/figures/`）
3. 创建单元测试和集成测试文件
4. 在 `app/agents/__init__.py` 中导出新函数（可选，取决于 graph 集成需求）
5. 无需修改 `main.py`、`workflow.py` 或任何 Schema 文件

**回滚**：删除新增文件即可完全回滚，不影响任何现有脚手架。

---

## Open Questions

1. **`Name` 列的删除时机**：目前决策是在特征工程阶段删除，但需要团队确认：清洗报告中是否需要明确说明该列"延迟删除"？
2. **图表数量**：spec 要求"至少一张图"，建议生成：缺失值热力图、`Survived` 各特征分布图共约 5 张，是否有数量上限？
3. **`PassengerId` 在测试集的处理**：spec 要求保留，特征工程阶段是否同样保留？（用于最终提交文件对齐）
