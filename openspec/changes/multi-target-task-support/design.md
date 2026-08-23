## 背景

当前流水线基于 LangGraph 构建，包含 Preliminary EDA → Data Cleaning → In-depth EDA → Feature Engineering → Model Building 五个核心阶段。`target_spec` 数据结构已在 `state_schema.py` 中定义并由 Reader 节点填充，但三个中游 Agent（Preliminary EDA、Data Cleaning、Feature Engineering）及其底层工具尚未完成迁移，仍存在以下硬编码问题：

1. `preliminary_eda.py` 第 53 行：`target_col = "Survived" if "Survived" in df_train.columns else None` —— 硬编码了 Titanic 数据集的目标列名
2. `profiler.py` 签名为 `target_col: str`，仅支持单目标
3. `data_cleaning.py` 和 `feature_engineering.py` 的清洗/特征工具不感知目标列，可能误删或编码目标列
4. Planner Prompt（`prompts.py`）的清洗和特征工程阶段未注入 `target_spec` 上下文，LLM 不知道应该保护哪些列

`reviewer_tools.py` 中的 `review_data_cleaning` 和 `review_feature_engineering` 已经基于 `target_spec` 进行了目标列保留/泄漏检查——这意味着 Reviewer 的验证标准已经到位，但上游 Agent 的实现还没跟上，导致审查经常失败。

## 目标 / 非目标

**目标：**
- 三个 Agent 节点统一从 `state.target_spec.target_columns` 读取目标信息
- `profiler.py` 支持多目标画像（多分类分布 / 回归连续统计 / 多标签共现）
- 清洗和特征工程工具自动保护目标列，防止误删和泄漏
- Planner Prompt 注入 `target_spec` 摘要，使 LLM 感知任务类型
- 通过 8 个验证数据集的端到端测试

**非目标：**
- 不修改 Reader 节点的目标推断逻辑（已实现）
- 不修改 Model Building 节点（已支持多任务类型）
- 不修改 `state_schema.py` 的结构定义（`TargetSpec` 已完备）
- 不修改 Reviewer 的验证逻辑（已基于 `target_spec` 工作）

## 设计决策

### 决策 1：Profiler 接口从 `target_col: str` 改为 `target_cols: list[str]`

**选择**：修改 `profile_dataset()` 签名，接受 `target_cols: list[str] | None`，内部根据目标数量和类型生成不同的画像内容。

**理由**：Profiler 是唯一的数据画像入口，所有后续 Planner 都依赖它的输出。如果不在源头支持多目标，就需要在每个消费者处做适配，违反 DRY 原则。

**替代方案**：保持单目标接口，在外层循环调用。被否决，因为多标签场景需要生成标签间的共现矩阵，无法通过简单循环实现。

**具体变更**：
- `target_cols` 为空时，通过启发式（train/test 差异列）尝试推断
- 分类目标：生成 `class_distribution`（各目标各自的类别分布字典）
- 回归目标：生成 `target_statistics`（均值、中位数、偏度、峰度）
- 多标签目标：额外生成 `label_cooccurrence`（标签间皮尔逊相关矩阵）

### 决策 2：Agent 通过辅助函数统一获取目标列

**选择**：在 `app/utils/target_scope.py` 中新增 `get_target_columns(state: GraphState) -> list[str]` 辅助函数，所有 Agent 通过此函数获取目标列。

**理由**：集中一个入口获取目标列，避免各 Agent 分别编写 fallback 逻辑。辅助函数内部优先读 `target_spec.target_columns`，fallback 到旧字段（`task_config.target_columns`、`data_state.target_columns`），保证向后兼容。

**替代方案**：直接在每个 Agent 中读 `state.target_spec.target_columns`。被否决，因为迁移期内可能有旧流程未填充 `target_spec`，需要统一的降级策略。

### 决策 3：清洗工具增加目标列保护层

**选择**：在 `data_cleaning.py` 的工具执行循环中，增加前置过滤逻辑：
1. 如果工具是 `drop_columns`，自动从参数 `columns` 中剔除目标列和 ID 列
2. 工具执行后，验证训练集目标列未丢失

**理由**：LLM Planner 可能因为目标列的缺失率或统计特征而建议删除目标列（如 `plate_defect` 的 7 个二值目标列的分布极度不均），在工具执行层面拦截比在 Prompt 约束更可靠。

**替代方案**：仅通过 Prompt 指示 LLM 不要删除目标列。被否决，因为 LLM 的遵循率无法保证 100%，且硬编码保护更安全。

### 决策 4：Planner Prompt 注入 `target_spec` 摘要

**选择**：修改 `build_data_cleaning_prompt()` 和 `build_feature_engineering_prompt()`，接受并注入一个 `target_spec_summary` JSON 块。

**理由**：LLM 需要知道哪些列是目标列才能做出合理的清洗和特征工程决策。`build_model_building_prompt()` 已经实现了这一模式，保持一致性。

**注入内容**：
```json
{
  "target_columns": ["Exited"],
  "task_family": "binary_classification",
  "target_mode": "single_target",
  "id_columns": ["id", "CustomerId"]
}
```

### 决策 5：Feature Engineering 的 `selected_features` 自动排除目标列和 ID 列

**选择**：在 `feature_engineering.py` 生成 `selected_features` 时，通过集合差运算排除 `target_spec.target_columns` 和 `target_spec.submission.id_columns`。

**理由**：`reviewer_tools.py` 的 `review_feature_engineering` 已经检查了这一点（第 274-291 行），如果 `selected_features` 包含目标列会直接 blocking failure。在源头排除比让 Reviewer 打回重试更高效。

## 风险 / 权衡

| 风险 | 缓解措施 |
|------|----------|
| `profiler.py` 签名变更可能影响其他调用方 | 使用 `target_cols: list[str] \| None = None` 默认参数，保持向后兼容。旧的 `target_col: str` 参数保留为 deprecated alias |
| LLM 仍可能在 Prompt 注入 target_spec 后做出错误的工具调用 | 工具执行层的硬编码保护（决策 3）作为最后防线；Reviewer 的验证逻辑已到位 |
| 多标签数据集（plate_defect，7 个目标）的画像 JSON 可能过大 | 多标签共现矩阵仅生成上三角，压缩 JSON 体积 |
| 回归数据集（house_prices）的 `class_distribution` 字段无意义 | Profiler 根据 `task_family` 判断，回归目标输出 `target_statistics` 而非 `class_distribution` |
| 部分旧测试用例可能依赖 `target_col: str` 的旧签名 | 在迁移期保留旧参数作为 alias，触发 deprecation warning |

## 开放问题

- `preliminary_eda.py` 生成的 EDA 可视化图表（缺失值柱状图、分布图等）是否需要为每个目标列分别生成？还是只生成一个聚合视图？
- 当 `target_spec.status == "unresolved"` 时，Agent 应该报错还是用启发式继续执行？当前倾向于用启发式（train/test 差异推断目标列），并在 `data_quality_flags` 中记录警告。
