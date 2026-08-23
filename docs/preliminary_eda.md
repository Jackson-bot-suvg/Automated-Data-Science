# Preliminary EDA

## 作用

`app/agents/preliminary_eda.py` 对原始 train 数据做第一轮确定性画像，给后续 cleaning、planner 和 reviewer 提供结构化 EDA 上下文。

## 读取

- `data_state.current_train_path`
- `data_state.current_test_path`
- `target_spec.target_columns`

## 写入

- `data_state.schema_summary`
- `data_state.eda_findings`
- `artifacts.preliminary_eda_report`
- `artifacts.figures`
- `current_phase = "preliminary_eda"`

## 当前实现

- 完全确定性，不调用 LLM。
- 底层使用 `app/tools/profiler.py::profile_dataset()`：
  - 分类目标生成类别分布
  - 回归目标生成统计量
  - 多标签目标生成 label co-occurrence
- 产出两类结果：
  - 结构化 JSON 风格画像，写入 `data_state.eda_findings`
  - 便于人工查看的 markdown 报告和基础图表
- 目标列全部从 `target_spec` 读取，不再硬编码 Titanic 或单目标列名。
