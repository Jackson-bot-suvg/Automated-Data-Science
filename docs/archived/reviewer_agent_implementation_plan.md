# Reviewer Agent MVP Implementation

本文档记录当前主调用链 `Reviewer` 的 MVP 实现。它替换早期规划版文档，目标是让不熟悉 LangGraph 的同学也能看懂 reviewer 从 node 到 agent、tools、prompt、service、state update 的完整调用链。

## 1. 当前结论

主调用链 reviewer 已按以下层级落地：

```text
reviewer_node(state)
  -> ReviewerAgent.run(state)
      -> reviewer_tools 做确定性硬检查
      -> prompts/reviewer_prompt.py 组装 comment prompt
      -> services/reviewer_service.py 调 LLM 生成 comment
      -> ReviewerAgent 汇总 review_state / runtime_state update
```

当前实现支持的 execution phase：

```text
preliminary_eda
data_cleaning
feature_engineering
model_building
```

`Reviewer` 的核心定位是质量门，不是执行器。它不会重跑 EDA、清洗、特征工程或建模；它只检查刚完成的 `current_phase` 是否留下了足够证据，然后写入 `review_state.latest_review_status`，由 `route_after_reviewer(state)` 决定走 `approved`、`retry` 还是 `aborted`。

## 2. 实际文件职责

| 文件 | 职责 |
| --- | --- |
| `app/graph/reviewer_node.py` | LangGraph-facing adapter。只接收 `GraphState`，调用 `ReviewerAgent().run(state)`，返回局部 state update。 |
| `app/graph/workflow.py` | 注册 `reviewer` node。当前已经把 `NODE_NAMES[8]` 接到 `reviewer_node`。 |
| `app/graph/routes.py` | `route_after_reviewer(state)` 只读取 `state.review_state.latest_review_status`：`approved -> approved`，`fatal -> aborted`，其他 -> `retry`。 |
| `app/agents/reviewer.py` | 兼容旧入口 `run_reviewer(state)`，内部调用 `ReviewerAgent().run(state)`。 |
| `app/agents/reviewer_agent.py` | 主实现。读取 `current_phase`，分派硬检查，计算 `approved/retry/fatal`，调用 LLM comment service，汇总 `review_state` 和 `runtime_state`。 |
| `app/tools/reviewer_tools.py` | 确定性硬检查。包括 artifact 存在性、CSV 可读性、清洗后 target 是否保留、feature train/test 列对齐、model submission 行数和列结构。 |
| `app/prompts/reviewer_prompt.py` | LLM comment prompt builder。只接收硬检查后的 evidence，不接收完整 CSV。 |
| `app/services/reviewer_service.py` | LLM service adapter。使用 `langchain_openai.ChatOpenAI` 生成自然语言 comment；如果未配置 key、关闭 LLM 或调用失败，返回 deterministic fallback comment。 |
| `app/schemas/reviewer_schema.py` | `ReviewCheckResult`、`ReviewRecord`、`ReviewStatus`，确保 reviewer 中间结果和 history record 可序列化。 |
| `tests/unit/test_reviewer_agent.py` | Reviewer MVP 单测，覆盖 approve、retry、fatal、LLM comment service 调用点和 LLM disabled fallback。 |

注意：仓库当前已有 `app/graph/nodes.py` 这个扁平文件，所以本次没有创建 `app/graph/nodes/reviewer_node.py` 目录，避免 Python module/package 冲突。

## 3. State 读写契约

Reviewer 读取：

```text
current_phase
task_config.target_column
task_config.dataset_paths.test
problem_understanding.target_column
data_state
feature_state
model_state
artifacts
review_state.retry_count_by_phase
review_state.pending_review_phase
runtime_state.workspace_root
```

Reviewer 写入：

```text
review_state.latest_review_status
review_state.review_history
review_state.retry_count_by_phase
review_state.fatal_errors
review_state.last_review_comment
review_state.pending_review_phase
runtime_state.interrupted
runtime_state.last_reviewed_phase
```

Reviewer 不新增任何 GraphState 顶层字段，符合 `docs/engineering-specification.md` 的 closed top-level state 规则。

## 4. 状态判定规则

硬检查结果由 `ReviewCheckResult` 表示：

```text
blocking_failures: 会阻止进入下一阶段
warnings: 不阻止进入下一阶段，但写进 reviewer comment 和 history
checked_artifacts: 本次检查过的 artifact key
checked_state_fields: 本次检查过的 state 字段
details: 可序列化补充信息，例如 CSV shape
```

状态计算规则：

```text
无 blocking_failures -> approved
有 blocking_failures 且本 phase 累计失败次数 < 3 -> retry
有 blocking_failures 且本 phase 累计失败次数 >= 3 -> fatal
```

失败时才更新 `review_state.retry_count_by_phase[phase]`。第三次失败会把错误追加进 `review_state.fatal_errors`，并设置 `runtime_state.interrupted=True`。

## 5. 每个 phase 的 MVP 硬检查

### preliminary_eda

检查：

```text
artifacts.preliminary_eda_report 存在
data_state.schema_summary 非空
data_state.eda_findings 非空
data_state.eda_findings 推荐包含 dataset_shape / columns
artifacts.figures 为空时只给 warning
```

### data_cleaning

检查：

```text
artifacts.cleaned_train_data 存在且 CSV 可读非空
artifacts.cleaned_test_data 存在且 CSV 可读非空
artifacts.cleaning_report 存在
data_state.current_train_path 指向 artifacts.cleaned_train_data
data_state.current_test_path 指向 artifacts.cleaned_test_data
清洗后训练集保留 target column
data_state.cleaning_summary 为空时 warning
data_state.data_quality_flags 为空时 warning
```

### feature_engineering

检查：

```text
artifacts.feature_summary 存在
feature_state.engineered_train_path 存在且 CSV 可读非空
feature_state.engineered_test_path 存在且 CSV 可读非空
feature_state.selected_features 非空
engineered train/test 去掉 target 后列集合一致
feature_state.feature_transform_summary 为空时 warning
feature_state.leakage_checks 为空时 warning
```

### model_building

检查：

```text
artifacts.model_results 存在
artifacts.best_model 存在
artifacts.prediction_file 存在且 CSV 可读非空
model_state.prediction_path 指向 artifacts.prediction_file
model_state.candidate_models 非空
model_state.best_model_name 非空且属于 candidate_models
prediction_file 包含 target column
Titanic/Survived 场景下 prediction_file 包含 PassengerId 或 index
prediction_file 行数等于 task_config.dataset_paths.test 或 feature_state.engineered_test_path 的行数
model_state.validation_metrics 为空时 warning
model_state.cv_results 为空时 warning
model_state.baseline_metrics 为空时 warning
```

## 6. LLM Review 能力边界

LLM review 已接入，但只负责自然语言审核意见：

```text
deterministic checks -> status
deterministic checks -> prompt evidence
LLM -> last_review_comment
```

LLM 不能覆盖 `approved/retry/fatal`，这是为了避免“comment 看起来通过，但硬检查失败”的不一致。

可用 `.env` 配置项：

```text
REVIEWER_LLM_ENABLED=false   # 关闭 LLM，直接使用 fallback comment
REVIEWER_LLM_API_KEY         # reviewer 专用 key，优先级最高
REVIEWER_LLM_MODEL           # 默认 glm-5
REVIEWER_LLM_BASE_URL        # reviewer 专用 base url
REVIEWER_LLM_TEMPERATURE     # 默认 0
REVIEWER_LLM_MAX_TOKENS      # 默认 512
```

如果 `REVIEWER_LLM_API_KEY` 留空，会依次尝试 `.env` 中的：

```text
DASHSCOPE_API_KEY
ZHIPUAI_API_KEY
OPENAI_API_KEY
```

如果没有 key 或调用失败，`services/reviewer_service.py` 会返回 fallback comment，不中断 LangGraph 主流程。

## 7. 当前仍未覆盖的 gap

MVP 已经能作为可运行 reviewer node 注入主图，但还有这些后续工作：

```text
1. in_depth_eda 还没有 phase-specific reviewer；当前若 current_phase=in_depth_eda 会 retry/fatal。
2. 还没有做人工中断点 interrupt/resume；fatal 只通过 runtime_state.interrupted=True 表示。
3. LLM comment 目前是纯文本，不要求结构化 JSON。
4. reviewer 没有读取 execution_plan.success_criteria 做逐项验收；当前只做 artifact/state 基线验收。
5. warning 策略较保守，后续可以按比赛类型或业务任务配置提高/降低严格度。
```

## 8. 验证

已添加并通过：

```text
pytest -q tests/unit/test_reviewer_agent.py
pytest -q tests/unit/test_architecture_scaffold.py
```

覆盖点：

```text
data_cleaning 正常产物 -> approved
feature_engineering train/test 列不一致 -> retry
model_building 第三次失败 -> fatal
run_reviewer 会调用 LLM comment service
LLM disabled 时 service 返回 fallback comment
```
