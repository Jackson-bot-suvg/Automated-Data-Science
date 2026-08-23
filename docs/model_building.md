# Model Building

## 作用

`app/agents/model_building.py` 基于 engineered features 和 `target_spec` 执行训练、选模、评估与 prediction 组装，是整条技术链路里耗时最长、合同最严格的阶段。

## 读取

- `target_spec`
- `feature_state.engineered_train_path`
- `feature_state.engineered_test_path`
- `feature_state.selected_features`
- `execution_plan.success_criteria["model_building"].agent_plan`

## 写入

- `model_state.candidate_models`
- `model_state.best_model_name`
- `model_state.best_params`
- `model_state.validation_metrics`
- `model_state.cv_results`
- `model_state.prediction_path`
- `model_state.baseline_metrics`
- `model_state.model_explanations`
- `feature_state.feature_importance_summary`
- `artifacts.model_results`
- `artifacts.best_model`
- `artifacts.prediction_file`
- `artifacts.phase_plan_records.model_building`
- `current_phase = "model_building"`

## 当前实现

- 目标合同解析以 `target_spec` 为主，legacy mirrors 只在极少数缺失场景兜底。
- 已支持的任务族 × target_mode：
  - `binary_classification × single_target`
  - `multiclass_classification × single_target`
  - `regression × single_target`
  - `regression × multi_output`（`MultiOutputRegressor` 包装）
  - `multilabel_classification × multi_label`，运行时按训练数据再细分：
    - `binary_multilabel`（每列 ⊆ {0, 1}，原 multilabel 路径）
    - `multiclass_multioutput`（≥1 列有 3+ 类别，使用 `make_scorer` 包的
      subset-accuracy 作为 refit、per-target macro-F1 平均作为 secondary，
      自动跳过 dummy baseline）
  - `clustering × single_target`（X-only，`KMeans` + 可选 `DBSCAN`，silhouette
    选优；DBSCAN 噪声点剔除后再算指标）
- 当前 phase plan 可以配置：
  - preprocessing
  - candidate models
  - CV
  - scoring
- planner 参数会先做正规化，再合并到 `ModelingConfig`：
  - dict-wrapped 值（如 `{"strategy": "median"}`）自动解包为内层原语；
  - 非法 impute 别名（`mode` / `average` / `zero`）映射到 `SimpleImputer`
    支持的策略；
  - `stratified_kfold` 在 multilabel / regression / clustering 上自动降级为
    `KFold`（`StratifiedKFold` 在这些 y 类型上会拒收）。
- 候选分类器默认 `class_weight='balanced'`（GB 不支持除外）；均衡数据下权重
  退化为 1.0，不平衡数据下提升少数类信号、不损 AUC。
- multilabel 任务的 `roc_auc_macro_ovr` primary metric 现在通过 `make_scorer`
  真正驱动 `GridSearchCV` 的 refit（之前因 sklearn 无原生 string scorer 支持
  multilabel-indicator 而偷偷退化为 `f1_micro`）。`cv_results` 暴露
  `cv_per_target_auc_mean / _std`。
- prediction 组装严格按 `target_spec.submission` 的列顺序执行；
  `multiclass_multioutput` 下 `prediction_kind="probability"` 自动回退到 label
  （per-label 正类概率对 K>2 列无意义）。
- baseline metrics 是 out-of-fold dummy baseline，仅在 task_family ≠
  clustering 且 subtype ≠ multiclass_multioutput 时计算。
- 全部候选失败时，每个候选的具体异常文本写入
  `data_state.data_quality_flags`，方便 reviewer / run_manifest 排查。
- phase 会同时产出：
  - best model artifact
  - model results
  - prediction file
  - feature importance / model explanation 摘要

## 备注

- `model_building` 通常是整条链路最耗时的节点，耗时主要来自 CV + candidate
  model search。当前对 plate_defect 有一处 DEV-ONLY 的 `cv.n_splits=2`
  覆盖（`run_model_building` 内、`task_config.dataset_root` 含 `plate_defect`
  时生效）；待后续 `configure_compute_budget` plan-step 接口落地后立即移除。
- 候选模型集合：分类器 LR / RF / GB / SVC / XGBoost；回归器 Ridge / Lasso /
  RF / GB / XGBoost；clustering KMeans / DBSCAN。XGBoost 走条件导入
  （强制安装），DBSCAN 同。
- **测试**：`test_model_tools.py` / `test_model_building.py` 等按 4 任务族 ×
  target_mode 端到端覆盖，并对 multi-target subtype、AUC scorer、CV 降级、
  impute 别名、`class_weight='balanced'` 默认值、clustering、`_error_envelope`
  保留 review_state 等关键路径单测覆盖。128 tests 全部通过。

## Interop Status

### Already compatible / does not block modeling 已可对接或不影响运行

| Module 模块                            | Note 备注                                                                                                       |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `app/agents/reader.py`                 | 已生成权威 `target_spec`（含 `task_family` / `target_mode` / `submission` / `evaluation`），同时保留 legacy mirrors |
| `app/agents/bg_reviewer.py`            | 完全读 `target_spec`；对 modeling 产出的 `submission_columns` 会正确校验                                             |
| `app/agents/indepth_eda.py`            | 已读 `target_spec.target_columns`                                                                              |
| `app/agents/preliminary_eda.py`        | 仍硬编码 Titanic `Survived`，但 modeling 只消费 engineered_train，不受影响                                           |
| `app/agents/data_cleaning.py`          | 不直接涉及 target；modeling 独立读 engineered 数据                                                                   |
| `app/agents/feature_engineering.py`    | 仍读旧 target mirrors，但产出 `engineered_train_path` / `selected_features` modeling 正常消费                       |
| `app/agents/planner.py`（主 Planner）   | modeling 走 `BasePlanner` 模式，不消费主 Planner 的 `agent_plan.steps`                                             |
| `app/tools/profiler.py`                | `profile_dataset()` 被 modeling 直接复用                                                                          |
| XGBoost                                | 强制安装                                                                                         |
