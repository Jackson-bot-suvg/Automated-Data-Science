# Planner

## 作用

Planner 现在只负责“为当前 phase 产出可执行计划”，不再直接决定整条 graph 的流程推进。流程推进由 graph + reviewer 结果控制。

## 读取

- `current_phase`
- `target_spec`
- 当前 phase 对应的数据文件
- `data_state.eda_findings`
- `data_state.cleaning_summary`
- `data_state.indepth_eda_summary`
- `feature_state.selected_features`

## 写入

- `execution_plan`
  - 其中当前 phase 的 `success_criteria[phase].agent_plan` 是正式执行计划

## 当前实现

- `planner_node` 先根据固定 phase 顺序和 reviewer 结果设置 `current_phase`：
  - `retry`：重跑当前 phase
  - `approved`：进入下一个 phase
- `PlannerAgent.run()` 只为这个 `current_phase` 生成计划，不再做 phase 选择。
- 计划生成分两层：
  - 先读取当前 phase 的真实数据上下文并构造 `profile_summary`
  - 再调用 phase-specific planner 生成 `ToolCallPlan`
- 如果 LLM 返回空步骤、非法参数或无效工具：
  - planner 会在内部做参数规范化和过滤
  - 仍为空时，planner 会生成一个确定性的默认 plan，而不是把空 plan 交给 phase agent

## 参数规范化

- 对 tool 参数做别名映射、字段过滤、必填校验
- 目前重点兼容：
  - `n_folds -> n_splits`
  - `primary -> primary_metric`
  - `secondary -> secondary_metrics`
  - `random_forest -> RandomForest`
  - `logistic_regression -> LogisticRegression`
  - `stratified -> stratified_kfold`

## 说明

- phase agent 默认直接执行 planner 写入的 `agent_plan`
- `phase_plan_records/*.json` 记录的是“实际执行的 phase plan”
- planner 本身不会写 `phase_history`，也不直接负责 summary / review
   - configure_scoring

中文：
Planner 会按阶段白名单过滤步骤，不在白名单中的工具会被丢弃。

## Task Normalization 任务归一化

EN:
Planner uses target_spec.task_family and target_spec.target_mode:
1. multilabel: multilabel classification or multi_label mode
2. multi_output: multi_output mode
3. classification: family contains classification
4. regression: family contains regression
5. default: fallback

中文：
Planner 基于 target_spec.task_family 与 target_spec.target_mode 做任务归一化：
1. multilabel
2. multi_output
3. classification
4. regression
5. default

## Flow-Control Update (Implemented) 流程控制更新（已实施）

EN:
1. Reviewer approved path now routes directly to explicit execution phases via graph routes.
2. Reviewer retry path now replays the same execution phase directly via graph routes.
3. Planner no longer drives phase transitions.

中文：
1. reviewer 的 approved 路径已通过 graph 路由直接进入明确的执行阶段。
2. reviewer 的 retry 路径已通过 graph 路由直接重跑当前执行阶段。
3. planner 不再驱动阶段跳转。

## This Round Changes 本轮改动

Updated:
1. app/graph/workflow.py
2. app/graph/routes.py
3. app/agents/planner.py
4. app/agents/indepth_eda.py
5. app/schemas/io_contracts.py
6. docs/planner.md

Planner-specific additions in this round:
1. Added strict phase tool parameter schemas into phase toollist prompt context.
2. Added planner-side alias mapping and parameter normalization for ToolCallPlan steps.
3. Added planner-side required-field gate; invalid steps are dropped before execution agents receive them.

本轮 Planner 重点新增：
1. 向 phase toollist prompt 注入严格参数 schema。
2. 在 planner 侧新增 ToolCallPlan 参数别名映射与规范化。
3. 在 planner 侧新增必填参数门禁；不合规 step 在下发前即被剔除。

Also updated tests (contract expectations):
1. tests/unit/test_planner_agent.py
2. tests/unit/test_planner_tracing.py
3. tests/unit/test_reader_bg_reviewer_summarizer.py

## Related Components 关联模块

1. app/agents/data_cleaning.py
2. app/agents/feature_engineering.py
3. app/agents/model_building.py
4. app/agents/indepth_eda.py
5. app/prompts/planner_prompt.py
6. app/graph/routes.py
7. app/graph/workflow.py
8. app/schemas/io_contracts.py

## Team Sync: Non-Planner Changes 联动改动（非 Planner 责任）

Purpose:
This section is for teammate alignment. The following updates are coordinated impacts of planner-side parameter normalization, but these modules are not responsible for planning decisions.

同步目的：
以下改动是配合 planner 参数规范化的联动影响，不代表这些模块承担规划职责。

1. app/agents/data_cleaning.py
   - Removed local parameter adaptation logic.
   - Now assumes planner emits normalized parameters.
   - Keeps target-column protection and execution logging responsibilities.

2. app/agents/feature_engineering.py
   - Removed local parameter adaptation logic.
   - Now assumes planner emits normalized parameters.
   - Keeps leakage protection, schema alignment, and execution logging responsibilities.

3. app/prompts/planner_prompt.py
   - Extended phase toollist prompt with strict per-tool parameter schema section.
   - Added explicit rules: exact field names, declared fields only, required fields must be present.

4. app/tools/cleaning_tools.py and app/tools/feature_tools.py
   - No behavior ownership change.
   - Function signatures remain source of truth for tool semantics; planner schemas must stay aligned with these signatures.

协作约定（建议）：
1. 新增/修改工具函数签名时，必须同步更新 planner.py 中对应 phase schema 与 alias 映射。
2. 如果出现新参数漂移样式，优先在 planner 侧修复，不在各 phase agent 重复实现。

## Retry Context and Repair 重试上下文与修复

Current state:
1. Reviewer already writes blocking_failures/details/comment into review_state.
2. Graph already handles retry routing.
3. Agent-side targeted auto-repair (feeding failures back into prompt) is partial and can be strengthened in a next step.

建议下一步：
1. 在各阶段 planner prompt 中显式注入最近一次该阶段 blocking_failures。
2. 对常见错误（schema mismatch/target leakage/missing artifact）增加规则化修复优先级。
