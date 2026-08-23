## 1. TaskIntake 与状态输入

- [x] 1.1 设计并实现 TaskIntake 层，用于把 `--task`、`--dataset`、可选 `--target` 转为初始 `GraphState`。
- [x] 1.2 在现有顶层 `task_config` 内保存或复用用户原始任务、数据集根目录和目标提示，避免新增 `GraphState` 顶层字段，也避免把目录路径硬塞进 `dataset_paths.train`。
- [x] 1.3 为 TaskIntake 增加单元测试，覆盖目录输入、可选 target、缺失 dataset 和非法路径。

## 2. Reader 目录理解与受控 ReAct

- [x] 2.1 扩展 Reader，使其在收到 dataset root 时能列举目录并识别 `train.csv`、`test.csv`、`sample_submission.csv` / `submission.csv`、`overview.txt`、`data_description.txt`。
- [x] 2.2 保持上下文有界：只读取有限 CSV metadata、小样本和有限长度文本说明，不读取完整大文件进 prompt。
- [x] 2.3 支持没有 `test.csv` 的情况，将其记录为 warning/risk，而不是直接 fatal。
- [x] 2.4 支持自然语言 target hint：当用户输入“我要预测客户是否流失”时，Reader 能结合列名推断具体 target column。
- [x] 2.5 增加 Reader 测试，覆盖标准 Kaggle 目录、只有 train 的目录、非标准 submission 文件名、target 自然语言补充和无法推断 target。

## 3. Background Reviewer 与人类确认

- [x] 3.1 调整 Background Reviewer 规则：train 缺失或不可读为 fatal；test/submission 缺失为 warning；target 不明确为 retry。
- [x] 3.2 在 CLI 中展示 Reader 理解结果，包括任务摘要、任务类型、目标列、数据文件、候选指标、风险和 reviewer comment。
- [x] 3.3 当 reviewer 返回 target 不明确的 retry 时，CLI 直接提示用户输入自然语言目标，并重新运行 Reader + Background Reviewer。
- [x] 3.4 当用户拒绝确认时，CLI 应干净退出，不进入 Planner。
- [x] 3.5 增加测试覆盖 approved、target retry、用户补充 target、用户拒绝确认和 fatal 路径。

## 4. main.py 主入口与完整 workflow

- [x] 4.1 新增 `main.py`，支持 `python main.py --task "用这个数据集预测客户流失" --dataset data/bank_churn`。
- [x] 4.2 支持 CLI 参数：`--task`、`--dataset`、`--target`、`--env-file`、`--recursion-limit`、`--yes`。
- [x] 4.3 用户确认后运行完整 `build_workflow(stop_after_technical_summary=False)`，默认生成 business report。
- [x] 4.4 输出最终 artifacts 路径：business report、technical report、prediction/submission 文件和主要中间产物。
- [x] 4.5 保留或复用现有 `run_multiagent.py` 的节点进度日志能力，避免重复实现大量日志逻辑。
- [x] 4.6 设置 `runtime_state.thread_id`，并在 graph 调用中传入 `configurable.thread_id`，满足持久化和后续人类确认恢复约定。
- [x] 4.7 增加 CLI 集成测试，至少覆盖 bank_churn 风格目录的 dry-run / mocked workflow 路径。

## 5. 只有 train.csv 的兼容

- [x] 5.1 梳理后续阶段对 `current_test_path` 的依赖，明确哪些阶段可以 test optional。
- [x] 5.2 对可兼容阶段做最小修改，允许仅 train 数据继续运行。
- [x] 5.3 如果某阶段暂不支持无 test 数据，应输出清晰的 fatal/retry 原因，而不是底层文件读取异常。
- [x] 5.4 在 technical/business report 中写明没有外部测试集的限制。
- [x] 5.5 同步调整 Reviewer 和 artifact 验收标准：无 test 数据时不得强制要求真正的 external submission 文件。

## 6. 文档与 skills 维护

- [x] 6.1 更新 `README.md` 和 `README_cn.md`，替换或补充真实 `python main.py` 用法。
- [x] 6.2 完善 `docs/task_intake_layer.md`，说明 TaskIntake、Reader ReAct、Background Reviewer、人类确认和 target 补充流程。
- [x] 6.3 更新 `docs/graph_design.md`，说明 CLI 默认跑完整 workflow 到 business report。
- [x] 6.4 检查并维护 `docs/skills/` 下与模块创建、代码评审、部署相关的技能文档，确保它们引用新的入口和职责边界。
- [x] 6.5 如发现归档文档与当前行为冲突，只补充必要说明，不做无关大规模重写。

## 7. 验证

- [x] 7.1 运行新增和受影响的单元测试。
- [x] 7.2 运行 Reader / Background Reviewer / Planner / Business Translator 相关聚焦测试。
- [x] 7.3 在本地环境可用时运行完整单元测试。
- [x] 7.4 手动验证示例命令可以生成 business report。
