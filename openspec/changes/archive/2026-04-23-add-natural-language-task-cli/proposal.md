## 为什么

当前项目已经具备 `Reader -> Background Reviewer -> Planner -> 执行阶段 -> Reviewer -> Technical Summarizer -> Business Translator` 的主链路，但用户入口仍然不完整：

- 仓库根目录没有 `main.py`，而 README、Dockerfile 和 docker-compose 都指向 `python main.py`。
- 现有 `run_multiagent.py` 只支持硬编码 Titanic 数据集，不能接受自然语言任务和任意数据集目录。
- `ReaderAgent` 已能基于结构化 `TaskConfig` 和 CSV 元数据生成任务理解，但还没有面向 CLI 的任务接入层，也没有人类确认环节。
- 当目标列无法确定时，系统缺少自然语言补充和再次理解的交互流程。

本次变更要把项目从“固定 demo 入口”推进到“用户可用的自然语言任务入口”：用户通过 CLI 输入任务描述和数据集目录，系统由 TaskIntake 初始化状态，由 Reader 读取目录和数据进行理解，由 Background Reviewer 审核理解结果，并在用户确认后跑完整 workflow 直到 business report。

## 改什么

- 新增 `main.py` 作为主入口，负责 CLI 参数、流程启动、人类确认和最终结果输出。
- 新增轻量 TaskIntake 层，负责把 `--task` 和 `--dataset` 转为初始 `GraphState` / `TaskConfig` 输入。
- 扩展 Reader 的输入语义，使 Reader 能根据数据集目录内容进行受控的 ReAct 式理解：
  - 读取目录文件列表。
  - 识别 train/test/submission/overview 等数据文件。
  - 读取有限 CSV 元数据和样本。
  - 推断 target、task type、候选指标和风险。
- 扩展 Background Reviewer / CLI 协作方式：
  - 审核 Reader 理解是否足够进入规划。
  - 将 Reader 理解结果展示给用户确认。
  - 当 target 不确定时，让用户用自然语言补充目标，再由 Reader 重新理解。
- 主入口默认运行完整图，最终生成 business report。
- 更新 `docs/` 内相关文档和 `docs/skills/` 说明，确保新入口、交互方式和开发约定可维护。

## 不改什么

- 不重写现有 LangGraph workflow 结构。
- 不让 LLM 直接生成和执行任意 Python 脚本。
- 不要求第一版支持 zip、xlsx、parquet、多表关系型数据集或自动训练/测试拆分。
- 不把用户确认做成 Web UI，本次仅实现 CLI 人类确认。

## 影响范围

- 新增或修改入口：`main.py`。
- 新增 TaskIntake 相关模块，建议位于 `app/intake/` 或同等清晰边界下。
- 可能扩展 `TaskConfig` 或新增 intake state/schema，用于保存 `raw_task` 与 `dataset_root`。
- 调整 Reader，使其能处理目录输入和用户补充目标的再次理解。
- 调整 Background Reviewer 或 CLI adapter，使 retry 时能够驱动用户补充。
- 更新 README、中文 README、图设计文档、任务接入文档和相关 skills 文档。

## 成功标准

- 用户可以运行：

```bash
python main.py --task "用这个数据集预测客户流失" --dataset data/bank_churn
```

- 系统能自动发现数据集目录中的 `train.csv`、可选 `test.csv`、可选 `sample_submission.csv` / `submission.csv`。
- 如果 target 可推断，系统展示理解结果并等待用户确认。
- 如果 target 不确定，系统允许用户用自然语言补充预测目标，并重新理解到具体列名。
- 如果没有 `test.csv`，系统以 warning 继续，并在报告中说明没有外部测试集的影响。
- 用户确认后，workflow 运行到 business report，并打印关键产物路径。
- `docs/` 和 `docs/skills/` 中的相关文档同步更新。
