# TargetSpec 迁移计划

## 目标

用 `state.target_spec` 统一表达单 target 和多 target。

迁移完成后，所有 agent 都只读 `state.target_spec`，不再从
`task_config`、`problem_understanding` 或 `data_state` 里读取 target 信息。

## 新的数据结构

在 `GraphState` 中新增：

```python
class GraphState(StrictBaseModel):
    task_config: TaskConfig = Field(default_factory=TaskConfig)
    target_spec: TargetSpec = Field(default_factory=TargetSpec)
    problem_understanding: ProblemUnderstanding = Field(default_factory=ProblemUnderstanding)
    ...
```

`TargetSpec` 是唯一的 target 协议：

```python
class TargetSpec(StrictBaseModel):
    status: str = "unresolved"          # unresolved / resolved / ambiguous

    user_target_hint: str = ""          # 用户输入的 target hint
    user_task_family_hint: str = ""     # 用户输入的 task type hint

    target_columns: list[str] = []      # 唯一权威 target 字段

    target_mode: str = "unknown"        # single_target / multi_label / multi_output
    task_family: str = "unknown"        # binary_classification / multilabel_classification / regression 等
    prediction_kind: str = "unknown"    # label / probability / continuous

    id_columns: list[str] = []
    raw_feature_columns: list[str] = []

    submission: SubmissionSpec = SubmissionSpec()
    evaluation: EvaluationSpec = EvaluationSpec()
    evidence: list[TargetEvidence] = []
```

提交协议：

```python
class SubmissionSpec(StrictBaseModel):
    sample_submission_path: str = ""
    id_columns: list[str] = []
    submission_columns: list[str] = []  # 最终 submission 的完整列顺序
    output_columns: list[str] = []      # 模型输出列，通常等于 target_columns
```

评估协议：

```python
class EvaluationSpec(StrictBaseModel):
    metric_hint: str = ""
    primary_metric: str = ""
    candidate_metrics: list[str] = []
```

证据记录：

```python
class TargetEvidence(StrictBaseModel):
    source: str = ""                    # user_hint / sample_submission / train_test_diff / overview
    columns: list[str] = []
    confidence: float = 0.0
    notes: str = ""
```

## 核心规则

`target_columns` 是唯一权威字段。

单 target：

```python
target_spec.target_columns = ["Survived"]
```

多 target：

```python
target_spec.target_columns = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]
```

不要再新增 `target_config`。`target_spec` 已经替代了所有 target 配置能力。

## 替换掉的旧字段

| 原字段 | 替换为 |
|---|---|
| `TaskConfig.target_column` | `target_spec.target_columns` |
| `TaskConfig.target_columns` | `target_spec.target_columns` |
| `TaskConfig.target_hint` | `target_spec.user_target_hint` |
| `TaskConfig.task_type` | `target_spec.user_task_family_hint` / `target_spec.task_family` |
| `TaskConfig.metric_hint` | `target_spec.evaluation.metric_hint` / `target_spec.evaluation.primary_metric` |
| `ProblemUnderstanding.target_column` | `target_spec.target_columns` |
| `ProblemUnderstanding.target_columns` | `target_spec.target_columns` |
| `ProblemUnderstanding.predicted_task_type` | `target_spec.task_family` |
| `ProblemUnderstanding.candidate_metrics` | `target_spec.evaluation.candidate_metrics` |
| `DataState.target_column` | `target_spec.target_columns` |
| `DataState.target_columns` | `target_spec.target_columns` |
| Planner `reader_outputs.target_column` | `target_spec` 摘要 |
| LangSmith metadata `target_column` | `target_spec` metadata |

## 迁移后的状态分工

### `TaskConfig`

只保存用户原始输入和数据入口，不再保存 target 信息。

```python
class TaskConfig(StrictBaseModel):
    task_id: str = ""
    raw_task: str = ""
    dataset_root: str = ""
    dataset_paths: DatasetPaths = Field(default_factory=DatasetPaths)
    business_context: str = ""
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
```

### `ProblemUnderstanding`

只保存人类可读的任务理解，不再保存 target 信息。

```python
class ProblemUnderstanding(StrictBaseModel):
    task_summary: str = ""
    data_overview: DataOverview = Field(default_factory=DataOverview)
    risks: list[str] = Field(default_factory=list)
    business_goal_summary: str = ""
```

### `DataState`

只保存数据文件和处理状态，不再保存 target 信息。

```python
class DataState(StrictBaseModel):
    raw_train_path: str = ""
    raw_test_path: str = ""
    current_train_path: str = ""
    current_test_path: str = ""
    schema_summary: dict[str, Any] = Field(default_factory=dict)
    cleaning_summary: dict[str, Any] = Field(default_factory=dict)
    eda_findings: dict[str, Any] = Field(default_factory=dict)
    indepth_eda_summary: dict[str, Any] = Field(default_factory=dict)
    data_quality_flags: list[str] = Field(default_factory=list)
    train_shape: tuple[int, int] | None = None
    test_shape: tuple[int, int] | None = None
```

## 迁移原则

1. Reader 负责生成 `target_spec`。
2. Reader 之后的 agent 只消费 `target_spec`。
3. Reviewer 用 `target_spec` 检查 target 是否丢失、泄漏或 submission schema 不匹配。
4. 旧 target 字段只作为短期兼容字段，后续删除。

## 各 Agent 迁移方向

### Task Intake

- 不再把 target 信息写成正式协议字段。
- 用户输入的 target、task type、metric hint 写入 `target_spec` 的 hint 字段。
- `task_config` 只保留原始任务、数据路径和业务上下文。

### Reader

- 负责生成完整 `target_spec`。
- 用 train/test 差异、sample submission、overview 和用户 hint 确认 target。
- 多 target 任务必须保留完整 `target_columns`。
- 填充 `target_mode`、`task_family`、`prediction_kind`、submission 协议和 evidence。

### Background Reviewer

- 检查 `target_spec` 是否 resolved。
- 检查所有 target columns 都在 train 中。
- 检查 test 中没有 target columns。
- 检查 sample submission 和 `target_spec.submission` 是否一致。

### Planner

- 从 `target_spec.task_family` 和 `target_spec.target_mode` 决定计划。
- 多 target 任务不能继续走单 target 的建模假设。
- planner prompt 和 state summary 需要显式包含 `target_spec`。

### EDA / In-depth EDA

- 统一读取 `target_spec.target_columns`。
- 单 target 做普通 target 分析。
- 多 target 做每个 target 的分布、共现关系和 feature-target 关系摘要。

### Data Cleaning

- train 输出必须保留所有 target columns。
- test 输出不能包含任何 target columns。
- 清洗逻辑只作用于 feature columns，避免 target 泄漏。

### Feature Engineering

- processed train 必须保留所有 target columns。
- processed test 不能包含 target columns。
- `selected_features` 不能包含 target columns 或 id columns。
- train/test feature schema 需要对齐。

### Modeling

- 单 target 使用 `target_columns[0]` 作为 y。
- 多 target 使用完整 `target_columns` 作为 Y。
- 多标签分类需要支持 multi-output / one-vs-rest / per-target models。
- prediction shape 必须匹配 submission output columns。

### Submission

- 按 `target_spec.submission.submission_columns` 输出。
- 不再自己猜 submission 列名。
- 多 target 时输出所有目标列。

### Reviewer

- 每个阶段都用 `target_spec` 验证 target 是否丢失或泄漏。
- 最终 submission 检查列名、列顺序、行数和空值。
- 多 target 任务不能因为旧的 `target_column` 为空而误判失败。

### Summarizer / Business Translator

- 报告中使用 `target_spec` 描述任务。
- 单 target 显示一个目标。
- 多 target 显示 target 数量、目标列和任务类型。

### Observability

- 日志和 LangSmith metadata 使用 `target_spec`。
- 记录 `target_count`、`target_mode`、`task_family`、`prediction_kind` 和主 metric。
