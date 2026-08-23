## 1. State Contract And Intake

- [x] 1.1 扩展入口相关 state schema，增加 `target_columns` 并保持单目标路径兼容
- [x] 1.2 调整 TaskIntake 的初始状态构造与辅助逻辑，使其保留多目标相关 target hint 而不提前压缩为单列

## 2. Reader And Reviewer

- [x] 2.1 重构 Reader target inference 顺序，优先使用 sample submission 与 train/test 差异识别多目标列集合
- [x] 2.2 更新 Reader 输出 payload、task summary 和风险信息，使其能够表达单目标或多目标 target scope
- [x] 2.3 更新 Background Reviewer 的 target 校验逻辑，使其在多目标列集合有效时返回 approved，并仅在 target scope 未解析时 retry

## 3. CLI Experience And Coverage

- [x] 3.1 更新 `main.py` 的确认展示与补充提问逻辑，支持显示和传递 `Target columns`
- [x] 3.2 增加或调整 CLI、Reader、Background Reviewer 单元测试，覆盖 `plate_defect` 风格多目标数据集
- [x] 3.3 更新 `docs/task_intake_layer.md`、`docs/reader.md` 和相关入口说明，反映多目标入口行为
