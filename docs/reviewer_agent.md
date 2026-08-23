# Reviewer Agent

## 作用

`app/agents/reviewer_agent.py` 在每个 execution phase 后做确定性审核，并决定后续是 `approved`、`retry` 还是 `fatal`。

## 读取

- `current_phase`
- `target_spec`
- 当前 phase 对应的 state
- 当前 phase 对应的 artifacts
- `phase_plan_records.*`

## 写入

- `review_state.latest_review_status`
- `review_state.review_history`
- `review_state.retry_count_by_phase`
- `review_state.fatal_errors`
- `review_state.last_review_comment`
- `review_state.pending_review_phase`
- `artifacts.phase_review_records.<phase>`
- `runtime_state.last_reviewed_phase`

## 当前实现

- `reviewer_tools.py` 是主要的确定性检查层。
- 审核覆盖：
  - `preliminary_eda`
  - `data_cleaning`
  - `indepth_eda`
  - `feature_engineering`
  - `model_building`
- Reviewer 已以 `target_spec` 为统一合同，兼容：
  - 单目标回归
  - 单目标二分类
  - 单目标多分类
  - multi-target / multilabel
- 现在 reviewer 会显式读取 `phase_plan_records/<phase>.json`，校验当前 phase 是否有可追踪的执行计划，而不再假设 planner 一定正常产出。
