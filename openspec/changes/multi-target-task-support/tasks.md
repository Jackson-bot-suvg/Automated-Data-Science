## 1. 基础设施：目标列辅助函数与 Profiler 改造

- [x] 1.1 在 `app/utils/target_scope.py` 中新增 `get_target_columns(state: GraphState) -> list[str]` 辅助函数，优先读 `target_spec.target_columns`，fallback 到旧字段
- [x] 1.2 修改 `app/tools/profiler.py` 的 `profile_dataset()` 签名：新增 `target_cols: list[str] | None` 参数，保留 `target_col: str` 作为 deprecated alias
- [x] 1.3 在 `profile_dataset()` 中实现分类目标画像：对每个目标列生成 `class_distribution`（类别占比字典）
- [x] 1.4 在 `profile_dataset()` 中实现回归目标画像：对连续目标列生成 `target_statistics`（均值、中位数、标准差、偏度、峰度）
- [x] 1.5 在 `profile_dataset()` 中实现多标签目标画像：当目标列 >= 2 个且均为二值列时，生成 `label_cooccurrence`（皮尔逊相关矩阵上三角）
- [x] 1.6 在 `profile_dataset()` 中实现自动判断：目标列 `nunique <= 20` 或 object/bool 类型走分类路径，否则走回归路径

## 2. Preliminary EDA Agent 迁移

- [x] 2.1 修改 `app/agents/preliminary_eda.py`：移除 `target_col = "Survived" if "Survived" in df_train.columns else None` 硬编码
- [x] 2.2 替换为调用 `get_target_columns(state)` 获取目标列列表
- [x] 2.3 将 `profile_dataset(df_train, target_col=target_col)` 改为 `profile_dataset(df_train, target_cols=target_cols)`
- [x] 2.4 验证 EDA 报告中的目标统计信息与 `target_spec.task_family` 一致（分类→类别分布，回归→连续统计）

## 3. Planner Prompt 注入 target_spec

- [x] 3.1 修改 `app/planners/prompts.py` 的 `build_data_cleaning_prompt()`：新增 `target_spec_summary: str` 参数，注入 `【目标协议 (TargetSpec Summary)】` 段落
- [x] 3.2 修改 `app/planners/prompts.py` 的 `build_feature_engineering_prompt()`：新增 `target_spec_summary: str` 参数，注入 `【目标协议 (TargetSpec Summary)】` 段落
- [x] 3.3 修改 `app/planners/base.py` 的 `plan_data_cleaning()` 签名：新增 `target_spec_summary: dict` 参数，传递给 Prompt 构建函数
- [x] 3.4 修改 `app/planners/base.py` 的 `plan_feature_engineering()` 签名：新增 `target_spec_summary: dict` 参数，传递给 Prompt 构建函数
- [x] 3.5 在 Prompt 中明确指示 LLM：「以下列为目标列和 ID 列，严禁在清洗/特征工程中删除或编码」

## 4. Data Cleaning Agent 迁移

- [x] 4.1 修改 `app/agents/data_cleaning.py`：使用 `get_target_columns(state)` 获取目标列
- [x] 4.2 在工具执行循环中增加目标列保护逻辑：执行 `drop_columns` 前，从参数 `columns` 中自动剔除目标列和 ID 列
- [x] 4.3 构建 `target_spec_summary` 字典并传递给 `planner.plan_data_cleaning(eda_findings, target_spec_summary)`
- [x] 4.4 清洗完成后验证训练集仍包含所有目标列，缺失时返回 `_error_envelope`
- [x] 4.5 清洗完成后验证测试集不包含目标列，若包含则删除并记录 `data_quality_flags` 警告

## 5. Feature Engineering Agent 迁移

- [x] 5.1 修改 `app/agents/feature_engineering.py`：使用 `get_target_columns(state)` 获取目标列和 ID 列
- [x] 5.2 在工具执行循环中增加目标列保护逻辑：执行 `encode_categorical` 前，检查 column 是否为目标列，若是则跳过
- [x] 5.3 构建 `target_spec_summary` 字典并传递给 `planner.plan_feature_engineering(fresh_profile, target_spec_summary)`
- [x] 5.4 修改 `selected_features` 生成逻辑：通过集合差运算排除 `target_columns` 和 `id_columns`
- [x] 5.5 新增训练集/测试集特征 Schema 对齐检查：不一致时在 `data_quality_flags` 中记录并尝试对齐
- [x] 5.6 新增 `leakage_checks` 填充：记录目标列排除和 ID 列排除的检查结果

## 6. 端到端验证

- [ ] 6.1 使用 `data/bank_churn` 运行完整流水线（二分类，单目标 `Exited`）
- [ ] 6.2 使用 `data/titanic` 运行完整流水线（二分类，单目标 `Survived`）
- [ ] 6.3 使用 `data/spaceship_titanic` 运行完整流水线（二分类，单目标 `Transported`）
- [ ] 6.4 使用 `data/classification_with_an_academic_success_dataset` 运行完整流水线（多分类，单目标 `Target`）
- [ ] 6.5 使用 `data/ghouls_goblins_and_ghosts_boo` 运行完整流水线（多分类，单目标 `type`）
- [ ] 6.6 使用 `data/obesity_risks` 运行完整流水线（多分类，单目标 `NObeyesdad`）
- [ ] 6.7 使用 `data/house_prices` 运行完整流水线（回归，单目标 `SalePrice`）
- [ ] 6.8 使用 `data/plate_defect` 运行完整流水线（多标签分类，7 个目标列）
- [ ] 6.9 验证每个数据集的 Reviewer 审查（`review_preliminary_eda`、`review_data_cleaning`、`review_feature_engineering`）均通过
