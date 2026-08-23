# In-Depth EDA

## 作用

`app/agents/indepth_eda.py` 在清洗后的训练集上执行更深一层的分析，生成结构化分析摘要、图表和 markdown 报告，供 feature engineering、summarizer 和 business translator 使用。

## 读取

- `data_state.current_train_path`
- `data_state.eda_findings`
- `target_spec`
- `execution_plan.success_criteria["indepth_eda"].agent_plan`

## 写入

- `data_state.indepth_eda_summary`
- `data_state.data_quality_flags`
- `artifacts.indepth_eda_report`
- `artifacts.figures`
- `artifacts.phase_plan_records.indepth_eda`
- `current_phase = "indepth_eda"`

## 当前实现

- phase 执行的是确定性分析工具：
  - `target_analysis`
  - `feature_feature_relationships`
  - `outlier_analysis`
  - `segment_analysis`
- 当前任务语义优先从 `target_spec.task_family` / `target_spec.target_columns` 读取。
- 每个步骤的执行结果都会落在 `indepth_eda_summary` 中，并生成对应图表。
- 如果 planner 没有给出可执行步骤：
  - 会保留一个 fallback 摘要
  - 同时把 fallback 原因写入 `data_quality_flags` 和 `phase_plans/indepth_eda.json`

## 产物

- `in_depth_eda.md`
- figures
- `phase_plans/indepth_eda.json`


