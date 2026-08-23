# Data Cleaning

## 作用

`app/agents/data_cleaning.py` 执行清洗阶段。它读取 planner 已经写好的当前 phase plan，调用确定性的清洗工具，并把清洗结果和执行记录写回 state 与 artifacts。

## 读取

- `data_state.current_train_path`
- `data_state.current_test_path`
- `data_state.eda_findings`
- `target_spec`
- `execution_plan.success_criteria["data_cleaning"].agent_plan`

## 写入

- `data_state.current_train_path`
- `data_state.current_test_path`
- `data_state.cleaning_summary`
- `data_state.data_quality_flags`
- `artifacts.cleaned_train_data`
- `artifacts.cleaned_test_data`
- `artifacts.cleaning_report`
- `artifacts.phase_plan_records.data_cleaning`
- `current_phase = "data_cleaning"`

## 当前实现

- 工具层是确定性的：`drop_columns`、`fill_missing`、`clip_outliers`
- target 列和 id 列属于 protected columns：
  - 不允许被 `drop_columns` 删除
  - 清洗后 train 必须保留全部 target
  - test 不允许保留 target
- phase 不再自己重新调 planner，只执行 `execution_plan` 中的当前 phase plan。
- 如果 planner 没给出可执行步骤，phase 会走确定性 fallback，并把原因写入：
  - `data_state.data_quality_flags`
  - `phase_plans/data_cleaning.json`

## 产物

- `cleaned_train.csv`
- `cleaned_test.csv`
- `cleaning_report.md`
- `phase_plans/data_cleaning.json`
