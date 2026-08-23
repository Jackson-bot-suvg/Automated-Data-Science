# Reader

## 作用

`app/agents/reader.py` 是整条 workflow 的任务理解入口。它负责发现数据文件、解析目标协议、生成第一版任务理解，并把后续阶段需要的基础上下文写入 `GraphState`。

## 读取

- `task_config.raw_task`
- `task_config.dataset_root`
- `task_config.target_hint`
- `task_config.dataset_paths`
- 兼容镜像：`task_config.target_column`、`task_config.target_columns`
- `runtime_state.workspace_root`

## 写入

- 权威目标协议：`target_spec`
- 发现后的文件路径：`task_config.dataset_paths`
- 初始任务理解：`problem_understanding`
- 原始/当前数据路径：`data_state.raw_*`、`data_state.current_*`
- 数据基本画像：`data_state.train_shape`、`data_state.test_shape`、`data_state.eda_findings`、`data_state.schema_summary`
- 兼容镜像：`task_config.target_column(s)`、`data_state.target_column(s)`
- `current_phase = "reader"`

## 当前实现

- 支持从目录中发现 `train.csv`、`test.csv`、`sample_submission.csv`、`overview.txt` 等常见文件。
- 目标解析以结构化证据优先：
  1. 已有 target hint / 已解析结果
  2. `sample_submission` 与 train/test schema 差异
  3. `overview.txt` / `data_description.txt`
  4. 原始任务文本兜底
- `target_spec` 是 Reader 的正式产出，包含：
  - `target_columns`
  - `task_family`
  - `target_mode`
  - `prediction_kind`
  - `submission`
  - `evaluation`
  - `evidence`
- Reader 已兼容单目标分类、单目标回归、多分类、多标签 multi-target。
- legacy `target_column` / `target_columns` 仍会镜像写回，但只是兼容字段，不再是正式 contract。
