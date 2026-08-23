# Business Translator

## 作用

`app/agents/business_translator.py` 把 technical report 和结构化模型结果转换成 stakeholder-friendly 的业务报告，并同时导出 markdown 与 PDF。

## 读取

- `report_state.technical_report_path`
- `model_state.validation_metrics`
- `model_state.baseline_metrics`
- `feature_state.feature_importance_summary`
- `model_state.model_explanations`
- `task_config.business_context`
- `report_state.limitations`
- `target_spec`

## 写入

- `report_state.business_report_path`
- `report_state.executive_summary`
- `report_state.recommended_actions`
- `artifacts.business_report`
- `artifacts.business_report_pdf`
- `artifacts.figures`
- `current_phase = "business_translation"`

## 当前实现

- 不是纯技术报告复述，而是“技术结果 -> 业务含义 -> 建议动作”的转换层。
- 优先使用结构化证据：
  - validation metrics
  - baseline metrics
  - feature importance
  - model explanations
  - technical report artifacts
- 会补生成业务图表，并把图表路径注册进 artifacts。
- 当前 full workflow 默认会执行到这个节点；`run_multiagent.py` 现在也默认包含它。
