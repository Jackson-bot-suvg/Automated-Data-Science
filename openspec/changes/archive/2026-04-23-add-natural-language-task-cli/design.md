## 背景

现有系统的核心链路已经清晰：所有 agent 通过共享 `GraphState` 协作，Reader 负责初始化 `problem_understanding` 和 `data_state`，Background Reviewer 负责进入 Planner 前的质量门禁。问题在于入口层仍是 demo 型的，`run_multiagent.py` 硬编码 Titanic，而 `main.py` 缺失。

本次设计目标是补齐“自然语言任务接入层”，而不是改写 agent 主链路。设计上保持 KISS 和 SRP：`main.py` 只做 CLI、流程控制和人类确认；TaskIntake 只做输入归一化；Reader 继续做任务和数据理解；Background Reviewer 继续做理解结果审核。

## 总体流程

```text
python main.py --task "用这个数据集预测客户流失" --dataset data/bank_churn

        │
        ▼
main.py
  - 解析 CLI 参数
  - 调用 TaskIntake 构造初始 GraphState

        │
        ▼
TaskIntake
  - 保存 raw_task
  - 保存 dataset_root
  - 仅复用或扩展现有 TaskConfig，不新增 GraphState 顶层字段
  - 不提前强行假设 train/test 文件名

        │
        ▼
ReaderAgent
  - 读取 dataset_root 目录
  - 识别数据文件
  - 读取有限 CSV 元数据
  - 推断 target/task_type/metrics
  - 生成 problem_understanding

        │
        ▼
BackgroundReviewerAgent
  - 审核 train 是否可读
  - 审核 target 是否明确
  - 审核 Reader 理解是否足够

        │
        ▼
main.py 人类确认
  - approved: 展示理解结果，用户确认后继续
  - retry: 让用户补充自然语言信息，例如预测目标
  - fatal: 打印错误并停止

        │
        ▼
完整 workflow
  - Planner
  - EDA / Cleaning / Feature Engineering / Modeling
  - Technical Summarizer
  - Business Translator
```

## 模块职责

### main.py

`main.py` 是用户入口，负责：

- 定义 CLI 参数：
  - `--task`：自然语言任务描述，必填。
  - `--dataset`：数据集目录或文件路径，必填。
  - `--target`：可选，允许用户直接给出目标列或自然语言目标。
  - `--env-file`：可选，默认 `.env`。
  - `--yes`：可选，跳过人工确认，仅适合自动化环境。
  - `--recursion-limit`：可选，传给 LangGraph。
- 调用 TaskIntake 构造初始 `GraphState`。
- 先运行 Reader 和 Background Reviewer 的理解/审核阶段。
- 将 Reader 理解结果以简洁中文展示给用户。
- 当 Background Reviewer 返回 retry 且问题是 target 不确定时，提示用户用自然语言补充目标。
- 用户确认后，运行完整 workflow 到 business report。
- 调用 LangGraph 时必须设置 `runtime_state.thread_id`，并传入 `config={"configurable": {"thread_id": ...}}`，为后续 durable execution 和人类确认恢复保留架构基础。
- 打印最终产物路径。

### TaskIntake

TaskIntake 是薄层，不做复杂推理，负责：

- 将 CLI 输入保存为结构化初始状态。
- 保留用户原始自然语言任务，避免过早丢失信息。
- 保留数据集目录 `dataset_root`，让 Reader 自己读取目录并判断实际情况。
- 若用户显式传入 `--target`，将其保存为目标提示，但不强制认为它一定是精确列名。

为避免污染 `dataset_paths.train` 的语义，建议新增明确字段，例如：

```text
TaskConfig.raw_task: str
TaskConfig.dataset_root: str
TaskConfig.target_hint: str
```

这些字段必须落在现有顶层 `task_config` 内，或复用现有字段表达同等语义；不得为 TaskIntake 新增 `GraphState` 顶层 key。若实现选择不扩展 schema，第一版可以将原始任务写入 `task_config.business_context`，将用户目标提示写入 `task_config.target_column` 或内部解析变量，但不能把 dataset root 伪装成 `dataset_paths.train`。

### ReaderAgent

Reader 保持“任务和数据理解”职责，但输入从“已知 train/test 路径”扩展为“可能只有 dataset_root”。

Reader 的受控 ReAct 行为是确定性工具链式行为，不是任意代码执行：

```text
Observe: dataset_root = data/bank_churn
Act: list directory
Observe: train.csv, test.csv, sample_submission.csv, overview.txt
Act: read bounded CSV metadata
Observe: train columns/test columns/shape/dtypes/missingness/sample
Act: infer target/task_type/metrics
Output: problem_understanding + data_state
```

第一版只支持 CSV 和目录文件发现：

- `train.csv` 是必须文件，缺失为 fatal。
- `test.csv` 是可选文件，缺失为 warning。
- `sample_submission.csv` / `submission.csv` 是可选文件，缺失为 warning。
- `overview.txt` / `data_description.txt` 可作为额外任务上下文，但必须限制读取长度。

### Background Reviewer

Background Reviewer 继续保持确定性审核权威：

- train 缺失或不可读：fatal。
- target 不明确：retry。
- target 不在训练集列中：retry 或 fatal，取决于用户是否已补充过多次。
- test 缺失：warning，不阻止继续。
- submission 缺失：warning，不阻止继续。
- LLM 只能辅助生成评论，不得覆盖确定性状态。

### 人类确认

在进入 Planner 前，CLI 必须展示 Reader 理解：

```text
任务理解:
- 任务摘要: 预测客户是否流失
- 任务类型: classification
- 目标列: Exited
- 训练集: data/bank_churn/train.csv
- 测试集: data/bank_churn/test.csv
- 候选指标: accuracy, f1, roc_auc
- 风险/警告:
  - target 是自动推断的
  - 部分列存在缺失值

是否继续运行完整 workflow 到 business report? [y/N]
```

当 target 不确定时：

```text
Reader 无法确定目标列。
请用自然语言描述你想预测什么，例如：“我要预测客户是否流失”。
```

Reader 重新结合用户补充和列名进行判断。如果仍无法确定，则展示候选列并要求用户输入精确列名。

## 只有 train.csv 时的行为

没有 `test.csv` 时系统可以继续，但必须降低承诺：

- 可以做 EDA、清洗、特征工程和建模。
- 模型评估只能基于训练集内部验证切分或交叉验证。
- 不能生成真正的外部测试集预测。
- business report 必须明确说明没有外部测试集，结论可信度低于有独立测试集的情况。

如果现有后续阶段暂时还不能处理空 test path，实施时需要做最小兼容：

- 优先让后续阶段接受 test optional。
- 如果某阶段确实必须 test，则由 Background Reviewer 或 Planner 给出清晰 fatal/retry，而不是在深层 pandas 读取时报错。
- 同步调整 Reviewer 和报告验收标准：没有 `test.csv` 时不得要求生成真正的 `submission.csv`，但必须在 technical/business report 中说明该限制。

## 文档和 skills 维护

本次变更必须同步维护文档，避免入口行为和开发约定脱节：

- README / README_cn：更新 `python main.py` 的真实用法。
- `docs/task_intake_layer.md`：补充 TaskIntake 的职责、输入输出和交互流程。
- `docs/graph_design.md`：说明 full graph 默认运行到 business report。
- `docs/skills/`：更新与模块创建、代码评审或部署相关的 skills，使它们知道新入口和新边界。
- 如实现过程中发现旧 archived 文档与当前入口冲突，应只在必要处补充说明，不大规模重写归档内容。

## 风险和取舍

- 风险：Reader 自动推断 target 错误。
  - 缓解：Background Reviewer 审核、人类确认、自然语言补充目标。
- 风险：只有 train.csv 时后续阶段不兼容。
  - 缓解：将 test 缺失作为 warning，但在后续阶段做 optional test 兼容或清晰报错。
- 风险：Reader ReAct 范围膨胀。
  - 缓解：第一版只允许目录列举、有限文本读取和有限 CSV metadata，不执行任意代码。
- 风险：文档不同步。
  - 缓解：把 docs 和 skills 维护列入验收需求和任务清单。

## 迁移计划

1. 新增 TaskIntake schema/helper，构造初始 `GraphState`。
2. 新增 `main.py` CLI 主入口。
3. 扩展 Reader 的 dataset_root 发现和自然语言 target hint 理解。
4. 扩展 Background Reviewer / CLI retry 协作。
5. 保证 CLI 调用 graph 时设置 `thread_id` 和 `configurable.thread_id`。
6. 让 full workflow 从 CLI 默认跑到 business report。
7. 补充单元测试、CLI 集成测试和文档。
