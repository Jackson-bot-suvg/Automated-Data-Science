# Model Building Refactor Notes

> 分支 / Branch: `feature/reviewer-qa`
> 日期 / Date: 2026-04-18

---

## 1. 目标 / Goal

采用 **「Agent 仅做编排，纯计算逻辑抽离到 `app/tools/`」** 的模式，将模型构建逻辑从 `app/agents/model_building.py` 中解耦，迁出到新文件 `app/tools/model_tools.py`。

重构前，模型构建 agent 内部耦合了以下逻辑：

- 特征类型自动检测（`_detect_feature_types`）
- sklearn 预处理器构造（`_build_preprocessor`）
- 候选模型与超参网格（`_get_candidate_models`）
- GridSearchCV 主循环 + 5-fold CV 评分（内联在 `run_model_building` 中）
- 提交文件构造（`_build_submission`）
- model_results.json / best_model.pkl / submission.csv 工件落盘（内联）

重构后，这些逻辑以纯函数形式暴露在 `app/tools/model_tools.py`，可被任何编排方（agent、脚本、测试、未来的其它 Planner）独立调用，且功能与 I/O 契约 **完全保持不变**。

> **注意**：目前未在 model_building 阶段加入 LLM Planner。候选模型与超参搜索网格采用硬编码设置，`GridSearchCV` 自身已承担参数空间搜索的「规划」职责，功能上类似 Planner，已能满足 MVP 目标。后续扩展时可加入 Planner 与更多候选模型，将参数选择设为可选，由 LLM 决定模型与超参组合。

---

## 2. 变更文件清单 / Changed Files

| 文件 | 动作 | 说明 |
|------|------|------|
| `app/tools/model_tools.py` | **新增** | 抽离后的纯函数工具集 |
| `app/agents/model_building.py` | **重写** | 瘦身为编排节点，委托 tools 完成实际工作；模块 docstring 移除对 `standalone/` 的引用 |
| `tests/unit/test_model_tools.py` | **新增** | 24 个单测，覆盖新 tool 所有函数 |
| `.gitignore` | **修改** | 追加 `standalone/` 排除规则，防止本地实验代码误提交 |

未改动但仍覆盖：

- `app/tools/metric_tools.py`（保持原样，仍由 agent 调用）
- `tests/unit/test_model_building.py`（仍测 `run_model_building` 端到端行为，全部通过）
- `tests/unit/test_metric_tools.py`

---

## 3. `app/tools/model_tools.py` 公共 API

所有函数均为纯函数，不依赖 `GraphState`，不读写 `state`。

| 函数 | 签名 | 职责 |
|------|------|------|
| `detect_feature_types` | `(X, selected_features=None) -> (num_features, cat_features)` | 按 dtype 分离数值列与类别列，兼容 `selected_features` 子集 |
| `build_preprocessor` | `(num_features, cat_features) -> ColumnTransformer` | 数值列: `SimpleImputer(median) + StandardScaler`；类别列: `SimpleImputer(most_frequent) + OneHotEncoder` |
| `get_candidate_models` | `() -> dict[str, {estimator, param_grid}]` | 返回 4 个候选分类器：LR / RF / GB / SVC |
| `fit_and_select_model` | `(X, y, preprocessor, candidates, cv_splits=5, random_state=42) -> (best_pipe, best_name, all_results)` | 对每个候选做 GridSearchCV + 5-fold CV（accuracy/f1/roc_auc），按 CV accuracy 选最佳 |
| `build_submission` | `(test_df, preds, target_column) -> DataFrame` | 有 `PassengerId` 时保留，否则生成 `index` 列 |
| `summarize_cv_results` | `(candidate_names, best_name, all_results) -> dict` | 整理为 `model_results.json` 的内容结构 |
| `save_model_artifacts` | `(best_pipe, summary, submission, paths...) -> {artifact_key: path}` | 写入 3 个保留工件路径，返回 artifact key → path 映射 |

**工件路径常量**（从 agent 移入 tools，命名保持不变）：

```python
MODEL_RESULTS_PATH = "artifacts/models/model_results.json"
BEST_MODEL_PATH    = "artifacts/models/best_model.pkl"
PREDICTION_PATH    = "artifacts/predictions/submission.csv"
```

遵循 `docs/engineering-specification.md §9` 的保留工件约定，未变更 `artifact_schema.py::DEFAULT_ARTIFACT_PATHS` 中的任何键。

---

## 4. `app/agents/model_building.py` 重构后形态

`run_model_building(state)` 仅负责：

1. 从 `GraphState` 读取路径与目标列
2. 路径 / 目标列校验（失败时返回 `_error_envelope`）
3. 读 CSV → `detect_feature_types` → `build_preprocessor` → `get_candidate_models`
4. 委托 `fit_and_select_model` 完成训练与选型
5. 委托 `build_submission` + `summarize_cv_results` + `save_model_artifacts` 完成落盘
6. 调用 `metric_tools.compute_metric_bundle` 计算训练集指标
7. 组装并返回 State 更新字典

返回字典的键与字段值、`ModelState` 写入字段、工件路径 **与重构前逐字段一致**。

---

## 5. 契约兼容性 / Contract Compatibility

| 契约表面 | 状态 |
|----------|------|
| `run_model_building(state: GraphState) -> dict` 签名 | 未变 |
| 返回 dict 顶层键 `{model_state, artifacts, current_phase, phase_history}` | 未变 |
| 错误返回 `{runtime_state.interrupted, review_state.fatal_errors}` 结构 | 未变 |
| `ModelState` 写入字段（`candidate_models` / `best_model_name` / `best_params` / `validation_metrics` / `cv_results` / `prediction_path` / `baseline_metrics`） | 未变 |
| `artifacts` 写入键（`model_results` / `best_model` / `prediction_file`） | 未变 |
| `app/graph/nodes/modeling_node.py` 的调用方式 | 未变，仍 `from app.agents.model_building import run_model_building` |
| `app/schemas/io_contracts.py::NODE_CONTRACTS["model_building"]` | 未变 |
| `reviewer_tools.py::review_model_building` 的硬检查点 | 未变 |

因此：**feature/reviewer-qa 分支原有的图编排、reviewer 硬检查、summarizer 读取 ModelState 的链路无需任何修改**。

---

## 6. 测试 / Tests

```bash
python -m pytest tests/unit/test_model_tools.py \
                 tests/unit/test_model_building.py \
                 tests/unit/test_metric_tools.py
```

结果：**57 passed**。

覆盖率（`pytest-cov`）：

| Module | Stmts | Miss | Cover |
|--------|-------|------|-------|
| `app/tools/model_tools.py` | 77 | 1 | **99%** |
| `app/agents/model_building.py` | 44 | 3 | **93%** |
| `app/tools/metric_tools.py` | 33 | 0 | **100%** |
| **TOTAL** | 154 | 4 | **97%** |

全部模块 ≥ 80%，符合 `docs/git.md` 合并门槛。`metric_tools.py` 新增 5 条测试覆盖原有"sklearn 缺失时手工 F1 fallback"与"roc_auc 异常兜底"分支：

- `test_metric_bundle_roc_auc_exception_returns_none`
- `test_f1_fallback_perfect` / `test_f1_fallback_no_positives_predicted`
- `test_f1_fallback_all_zero_denom` / `test_f1_fallback_partial`

fallback 分支通过 `monkeypatch.setattr(builtins, "__import__", ...)` 模拟 `from sklearn.metrics import f1_score` 抛 ImportError 来触发。

新测试 `test_model_tools.py` 覆盖点：

- `detect_feature_types`：全数值 / 含类别 / 布尔列 / `selected_features` 子集 / 子集含不存在列 / 空子集回退
- `build_preprocessor`：仅数值 / 仅类别 / 混合（NaN 兜底）/ 全空
- `get_candidate_models`：键集合 / 结构 / 实例独立
- `fit_and_select_model`：正常选型 / 指标字段完整 / `classifier__` 前缀被剥离 / 最佳 pipeline 可预测 / 全候选失败时返回 `(None, "", {error})`
- `build_submission`：有 `PassengerId` / 无 `PassengerId`
- `summarize_cv_results`：常规汇总 / 空最佳名回退
- `save_model_artifacts`：三文件落盘 + JSON 内容 + pkl 可加载 / 保留路径常量值校验

---

## 7. 后续 / Next Steps

- 如需引入 LLM Planner：在 `BasePlanner` 新增 `plan_model_building(profile) -> ModelBuildingPlan`，由 LLM 基于数据画像决定候选模型清单与超参网格，将参数选择设为可选；`get_candidate_models()` 保留为 fallback，与 `data_cleaning` / `feature_engineering` 的 Planner 模式对齐
- 如需加入 XGBoost / LightGBM，只需在 `get_candidate_models()` 中追加条目
- 如需 SHAP 解释，可新增 `explain_model(pipeline, X) -> dict` 并写入 `ModelState.model_explanations`
- 如需替换选型指标（如改用 f1），在 `fit_and_select_model` 中调整 `best_score` 比较字段即可；注意同步更新 reviewer 硬检查
