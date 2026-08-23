# Background Reviewer

## 作用

`app/agents/bg_reviewer.py` 是 Reader 之后的入口 gate。它不负责修复问题，只负责判断 Reader 给出的任务理解和目标协议是否足以进入主 workflow。

## 读取

- `problem_understanding.task_summary`
- `problem_understanding.candidate_metrics`
- `target_spec`
- `data_state.current_train_path`
- `data_state.current_test_path`
- `runtime_state.workspace_root`

## 写入

- `review_state.latest_review_status`
- `review_state.review_history`
- `review_state.retry_count_by_phase`
- `review_state.fatal_errors`
- `review_state.last_review_comment`
- `review_state.pending_review_phase`
- `runtime_state.interrupted`
- `runtime_state.last_reviewed_phase`

## 当前实现

- `target_spec` 是这里唯一权威的目标协议来源。
- 主要检查：
  - train 是否存在且可读
  - `target_spec.status` 是否已 `resolved`
  - 所有 target 列是否存在于 train
  - test 是否误带 target 列
  - sample submission 列顺序是否匹配 `target_spec.submission.submission_columns`
- 输出规则：
  - 明确缺失或不可恢复问题：`fatal`
  - 目标未解析或 schema 不一致：`retry`
  - 满足合同：`approved`
- 已支持 multi-target，不会再强制把多目标任务压回单 target。
