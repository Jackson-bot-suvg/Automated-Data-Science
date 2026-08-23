# Summarizer

## 作用

`app/agents/summarizer.py` 在 `model_building` 审核通过后生成最终 technical report。它关注的是任务本身的技术结果，不是系统执行日志。

## 读取

- `target_spec`
- `task_config`
- `problem_understanding`
- `data_state`
- `feature_state`
- `model_state`
- `review_state`
- `artifacts`

## 写入

- `report_state.technical_report_path`
- `report_state.executive_summary`
- `report_state.limitations`
- `artifacts.technical_report`
- `current_phase = "technical_summary"`

## 当前实现

- 只有在：
  - `current_phase == "model_building"`
  - 且 reviewer `approved`
  时才真正产出报告。
- 报告以结构化 state/artifacts 为主，不依赖上游 markdown 反推。
- `target_spec` 是 target contract 的首选来源，legacy mirrors 仅兜底。
- 生成内容包括：
  - 问题定义
  - 数据摘要
  - 数据准备摘要
  - 特征工程摘要
  - 建模结果
  - prediction outputs
  - business translation inputs
  - artifacts
  - limitations
