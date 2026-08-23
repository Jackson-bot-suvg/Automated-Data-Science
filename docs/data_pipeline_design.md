# 数据流处理架构设计图与机制说明
(EDA -> Data Cleaning -> Feature Engineering)

本文档详述 AutoKaggle 核心数据演进管道（Data Pipeline）的设计哲学、组件交互及具体代码实现细节。架构采用**"大模型结构化决策中枢 (Planner) + 确定性工具调用箱 (Tool Calling)"**的范式，并通过 `target_spec` 实现多目标、多分类及回归任务的统一支持。

## 1. 核心设计哲学：决策与执行分离

在原有的传统架构中，大语言模型尝试直接阅读数据并直接返回写好的处理脚本，这种方式会导致：
- 幻觉带来不存在的数据列操作。
- `exec()` 注入带来的安全风险和执行期崩溃。
- 测试不可及（数据与模型强耦合）。

**当前架构的核心机制**：
- **感知器 (Profiler)**: 通过 pandas 操作，提取原始数据的统计学画像（含多目标类型自适应），并浓缩成 JSON。
- **决策大脑 (Planner / LLM)**: 接收 JSON 画像 + `target_spec_summary`，输出 `ToolCallPlan` JSON 对象。
- **执行手脚 (Tools)**: 预定义的安全数据处理工具（`app/tools/*_tools.py`），已适配 Pandas 3.x `StringDtype`。
- **流程阀门 (Agent Nodes)**: 调度上述组件，更新 `GraphState`，并实施目标列保护。

---

## 2. 目标列统一管理：target_spec 架构

### 2.1 核心辅助函数
**📍 源码**: `app/utils/target_scope.py`

所有 Agent 通过两个函数获取受保护列，确保全局一致：

```python
from app.utils.target_scope import get_target_columns, get_id_columns

target_cols = get_target_columns(state)  # → ['Exited'] / ['SalePrice'] / ['Pastry', ...]
id_cols = get_id_columns(state)          # → ['id'] / ['PassengerId'] / ['Id']
```

**Fallback 链**：`target_spec.target_columns` → `task_config.target_columns` → `task_config.target_column` → 空列表

### 2.2 支持的任务类型

| 任务类型 | 示例数据集 | target_spec 特征 |
|---------|-----------|-----------------|
| 二分类 | bank_churn, titanic, spaceship_titanic | 单目标，`task_family=classification` |
| 多分类 | academic_success, ghouls, obesity_risks | 单目标（可能是 `str` dtype），`task_family=classification` |
| 回归 | house_prices | 单目标数值列，`task_family=regression` |
| 多标签分类 | plate_defect | 7 个二值目标列，`target_mode=multi_label` |

---

## 3. 三大核心阶段解析

### 3.1 Preliminary EDA (数据侧写节点)
**📍 源码**: `app/agents/preliminary_eda.py`

**节点职责**：
数据处理管道的第一道入口，仅负责"纯读取与统计"。

**工作流**：
1. **输入获取**：从 `state.data_state.current_train_path` 拉取原始训练数据。
2. **目标列获取**：通过 `get_target_columns(state)` 读取目标列列表。
3. **生成画像 (Profile)**：调用 `profile_dataset(df, target_cols=[...])`，自动判断每个目标列类型：
   - **分类目标** → `class_distribution`（各类别占比）
   - **回归目标** → `target_statistics`（均值/中位数/标准差/偏度/峰度）
   - **多标签目标** → `label_cooccurrence`（标签共现皮尔逊相关矩阵）
4. **副作用落地 (Artifacts)**：
   - 可视化图表存放在 `artifacts/figures/`。
   - Markdown 报告 (`artifacts/eda/preliminary_eda.md`)。
5. **状态更新**：将 `eda_findings` JSON 塞入 `state.data_state.eda_findings` 下推。

---

### 3.2 Data Cleaning (数据清洗节点)
**📍 源码**: `app/agents/data_cleaning.py`

**节点职责**：
根据 EDA 画像处理缺失值、剔除无用列和截断异常值，同时实施目标列保护。

**工作流**：
1. **加载上下文**：读取 `eda_findings` 和原始数据集，通过 `get_target_columns(state)` 获取保护列。
2. **大模型动态规划 (Planning)**：
   - 调用 `plan_data_cleaning(eda_findings, target_spec_summary)`。
   - `target_spec_summary`（含目标列名、ID 列名、任务类型）被注入 Prompt，明确告知 LLM **硬约束：不得删除或修改受保护列**。
   - LLM 返回 `ToolCallPlan` 对象。
3. **目标列保护执行**：
   - 遍历工具调用前，自动从 `drop_columns` 参数中剔除受保护列。
   - 跳过并记录被拦截的操作到 `data_quality_flags`。
4. **清洗后验证**：
   - 训练集必须仍包含所有目标列（缺失触发 `_error_envelope`）。
   - 测试集不应包含目标列（包含则删除并记录警告）。
5. **输出**：`cleaned_train.csv` + `cleaning_report.md`。

**可用工具**：

| 工具 | 说明 | Pandas 3.x 适配 |
|------|------|-----------------|
| `drop_columns` | 删除列（受目标列保护拦截） | — |
| `fill_missing` | 填充缺失值 | `str` dtype 列 `mean`/`median` 自动降级为 `mode` |
| `clip_outliers` | 盖帽法截断异常值 | 非数值列静默跳过 |

---

### 3.3 Feature Engineering (特征工程节点)
**📍 源码**: `app/agents/feature_engineering.py`

**节点职责**：
编码分类特征、分箱连续特征、创建交互特征，并生成最终 `selected_features`。

**工作流**：
1. **状态承接**：读取清洗后的数据集 (`cleaned_train.csv`)。
2. **实时二次画像**：使用 `profile_dataset` 对最新数据重新画像。
3. **大模型规划**：
   - 调用 `plan_feature_engineering(fresh_profile, target_spec_summary)`。
   - 同样注入目标列保护的硬约束。
4. **目标列保护执行**：
   - `encode_categorical` / `create_binning`：若目标列参与则跳过。
   - `create_interaction`：若非数值列参与则跳过。
5. **selected_features 生成**：
   - `all_columns - target_columns - id_columns` 集合差运算。
6. **Schema 对齐**：
   - 训练集和测试集的特征列自动对齐（缺失补零，冗余删除）。
7. **泄漏检查 (leakage_checks)**：
   - 验证目标列和 ID 列是否已从 `selected_features` 中排除，写入 `data_quality_flags`。
8. **输出**：`engineered_train.csv` + `feature_summary.json`。

**可用工具**：

| 工具 | 说明 | Pandas 3.x 适配 |
|------|------|-----------------|
| `encode_categorical` | label / onehot / ordinal 编码 | onehot 输出 `int8`，受目标列保护 |
| `create_binning` | 分箱（quantile / uniform） | 失败回退填 0，受目标列保护 |
| `create_interaction` | 特征交互 +/-/*// | 非数值列自动跳过 |
| `extract_regex` | 正则提取文本特征 | — |
| `group_rare_categories` | 低频类别合并 | 使用 `.where()` 兼容 `StringDtype` |

---

## 4. 代码级的基建强耦说明

### A. 强类型的 Tool Calling Schema
**📍 源码**: `app/schemas/tools.py`

采用 `pydantic.BaseModel` 严格约束 LLM 输出：
```python
class ToolCall(BaseModel):
    tool: str
    parameters: Dict[str, Any]
    reasoning: Optional[str]

class ToolCallPlan(BaseModel):
    plan_name: str
    steps: List[ToolCall]
```

### B. 具备 target_spec 感知的 Planner
**📍 源码**: `app/planners/base.py` & `app/planners/prompts.py`

Planner 接口已扩展为接受 `target_spec_summary` 参数：
```python
plan = await planner.plan_data_cleaning(eda_findings, target_spec_summary)
plan = await planner.plan_feature_engineering(fresh_profile, target_spec_summary)
```

在 Prompt 模板中，`target_spec_summary` 被格式化后注入系统消息，形成 LLM 的硬约束：
- 目标列列表及任务类型
- 明确禁止删除/编码/分箱目标列的指令
- ID 列保护指令

### C. 确定型动作的工具箱
**📍 源码**: `app/tools/cleaning_tools.py` & `app/tools/feature_tools.py`

所有工具均已完成 **Pandas 3.x 兼容性适配**：
- 使用 `pd.api.types.is_string_dtype()` 替代 `dtype == "object"` 判断
- `fill_missing` 对 `str` dtype 列自动将 `mean`/`median` 降级为 `mode`
- `create_interaction` 拒绝字符串列参与运算
- `encode_categorical` onehot 输出强制 `int8`

### D. 端到端验证
**📍 源码**: `tests/test_pipeline_no_model.py`

覆盖 8 个数据集的自动化集成测试，验证：
- Reader 正确识别目标列
- Profiler 正确区分分类/回归/多标签
- 清洗阶段目标列存活
- 特征工程阶段目标列排除
- `selected_features` 不含任何受保护列
