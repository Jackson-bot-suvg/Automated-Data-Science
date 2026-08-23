## 为什么要做这个改动

当前流水线仅支持二分类任务（如 Titanic 的 `Survived`）。三个核心 Agent 节点——Preliminary EDA、Data Cleaning、Feature Engineering——存在硬编码的单目标假设，且没有从权威的 `target_spec` 读取目标信息。这导致系统无法运行在多分类（academic success、obesity risks、ghouls）、回归（house prices）和多标签分类（plate defect，7 个目标列）数据集上。作为 `target_spec` 迁移的一部分，这些 Agent 必须统一从 `state.target_spec` 消费目标信息，并正确处理所有任务类型。

## 改动内容

- **BREAKING**：Preliminary EDA、Data Cleaning、Feature Engineering 不再从 `task_config`、`problem_understanding` 或 `data_state` 的旧字段读取 target 信息，统一改为从 `state.target_spec` 读取。
- **Preliminary EDA**（`app/agents/preliminary_eda.py` + `app/tools/profiler.py`）：
  - 移除硬编码的 `target_col = "Survived"` 假设
  - 从 `state.target_spec.target_columns`（`list[str]`）读取目标列，支持多目标
  - `profile_dataset()` 更新为接受多个目标列，为分类目标生成各目标的类别分布，为回归目标生成连续值统计
  - 生成目标感知的 EDA 可视化图表（按目标的分布图、多标签的关联热力图）
- **Data Cleaning**（`app/agents/data_cleaning.py` + `app/tools/cleaning_tools.py`）：
  - 保护所有目标列不被删除或转换（防止目标泄漏）
  - 确保测试集输出不包含目标列
  - 验证训练集在清洗后仍保留所有目标列
  - 新增 target-safe 清洗工具：`safe_drop_columns` 自动排除目标列
- **Feature Engineering**（`app/agents/feature_engineering.py` + `app/tools/feature_tools.py`）：
  - `selected_features` 必须排除所有目标列和 ID 列（从 `target_spec` 读取）
  - 训练集/测试集的特征 Schema 对齐验证
  - 编码工具不得对目标列进行编码
  - 支持回归场景的特征转换（不依赖分类分布的逻辑处理连续目标）
- **Profiler**（`app/tools/profiler.py`）：
  - 接受 `target_cols: list[str]` 替代 `target_col: str`
  - 为连续目标生成回归感知的统计量（偏度、峰度）
  - 为多标签目标生成标签共现矩阵
- **Reviewer 工具**（`app/tools/reviewer_tools.py`）：
  - 更新 `review_preliminary_eda`、`review_data_cleaning`、`review_feature_engineering` 以基于 `target_spec` 进行验证
  - 新增目标泄漏检查：确保目标列不在测试数据和 `selected_features` 中

## 能力 (Capabilities)

### 新增能力
- `multi-target-profiling`：数据集画像支持多目标（列列表）、多分类分布和回归感知的连续值统计
- `target-safe-cleaning`：数据清洗工具和 Agent 保护目标列不被意外删除/转换，防止测试集目标泄漏
- `target-aware-feature-engineering`：特征工程 Agent 从特征选择中排除目标/ID 列，支持回归适用的特征转换，并验证训练集/测试集的 Schema 对齐

### 修改的已有能力
- `llm-planner`：BasePlanner 的 Prompt 必须注入 `target_spec` 上下文，使 LLM 知道当前是二分类、多分类、回归还是多标签任务

## 影响范围

- **Agent 代码**：`app/agents/preliminary_eda.py`、`app/agents/data_cleaning.py`、`app/agents/feature_engineering.py`
- **工具代码**：`app/tools/profiler.py`、`app/tools/cleaning_tools.py`、`app/tools/feature_tools.py`
- **审查代码**：`app/tools/reviewer_tools.py`（阶段审查函数）
- **Planner Prompt**：`app/planners/prompts.py`（注入 target_spec 上下文）
- **状态 Schema**：无需修改——`TargetSpec` 和多目标字段已存在于 `state_schema.py`
- **验证数据集**：`bank_churn`（二分类）、`academic_success`（多分类）、`ghouls`（多分类）、`house_prices`（回归）、`obesity_risks`（多分类）、`plate_defect`（多标签，7 个目标）、`spaceship_titanic`（二分类）、`titanic`（二分类）
