# Feature Engineering

## 作用

`app/agents/feature_engineering.py` 在清洗结果之上执行特征变换，生成可建模的 engineered train/test 和 `selected_features`。

## 读取

- `data_state.current_train_path`
- `data_state.current_test_path`
- `target_spec`
- `execution_plan.success_criteria["feature_engineering"].agent_plan`

## 写入

- `feature_state.engineered_train_path`
- `feature_state.engineered_test_path`
- `feature_state.selected_features`
- `feature_state.dropped_features`
- `feature_state.feature_transform_summary`
- `feature_state.leakage_checks`
- `artifacts.feature_summary`
- `artifacts.phase_plan_records.feature_engineering`
- `current_phase = "feature_engineering"`

## 当前实现

- 支持工具：
  - `encode_categorical`
  - `create_binning`
  - `create_interaction`
  - `extract_regex`
  - `group_rare_categories`
- target 列和 id 列是 protected columns：
  - 不允许进入 `selected_features`
  - 不允许作为需要编码/分箱/交互的输入输出列
- train/test feature schema 会在落盘前自动对齐：
  - test 缺失列补零
  - 多余列删除
- 如果 planner plan 不可执行，会在 phase 内记录 fallback 信息，但仍尽量保持输出合同完整。

## 产物

- `engineered_train.csv`
- `engineered_test.csv`
- `feature_summary.json`
- `phase_plans/feature_engineering.json`
