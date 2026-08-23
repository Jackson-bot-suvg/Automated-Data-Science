# Graph Design

## 当前主链路

```text
START
-> reader
-> background_reviewer
-> planner
-> preliminary_eda
-> reviewer
-> planner
-> data_cleaning
-> reviewer
-> planner
-> indepth_eda
-> reviewer
-> planner
-> feature_engineering
-> reviewer
-> planner
-> model_building
-> reviewer
-> summarizer
-> business_translator (full workflow only)
-> END
```

## 职责分工

- Graph：
  - 控制 phase 顺序
  - 根据 reviewer 结果决定 retry / next / end
- Planner：
  - 只为当前 phase 生成 `execution_plan`
- Phase agents：
  - 执行 planner 已写好的 phase plan
- Reviewer：
  - 决定当前 phase 是否通过
- Summarizer / Business Translator：
  - 负责最终技术报告和业务报告

## 当前路由规则

- `background_reviewer`
  - `approved -> planner`
  - `retry -> reader`
  - `fatal -> END`
- `reviewer`
  - `approved` 且 phase 不是 `model_building` -> `planner`
  - `approved` 且 phase 是 `model_building` -> `summarizer`
  - `retry -> planner`
  - `fatal -> END`
- `summarizer`
  - technical-only run：`END`
  - full workflow：`business_translator -> END`

## 当前脚本行为

- `main.py`：默认 full workflow
- `run_multiagent.py`：现在默认也跑 full workflow
- 如需 technical-only，可在 `run_multiagent.py` 使用 `--stop-after-technical-summary`
