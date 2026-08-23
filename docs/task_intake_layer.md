# Task Intake Layer

## 作用

`app/intake/task_intake.py` 是 CLI 的轻量入口层。它只负责校验输入、初始化 `GraphState` 和运行时身份，不负责解析目标协议。

## 写入

- `task_config.raw_task`
- `task_config.dataset_root`
- `task_config.target_hint`
- 兼容镜像：`task_config.target_column(s)`
- `task_config.dataset_paths`（仅在 dataset 是单个 CSV 文件时提前填 train）
- `runtime_state.thread_id`
- `runtime_state.run_id`
- `runtime_state.artifact_root`
- `artifacts`（按 run 隔离的一整套路径）

## 当前实现

- `thread_id` 是稳定的逻辑 workflow identity
- `run_id` 是每次执行唯一 identity
- run 级 artifacts 路径都从 `run_id` 派生
- 如果 dataset 输入是目录：
  - 文件发现交给 Reader
- 如果 dataset 输入是单个 CSV：
  - 直接写入 `task_config.dataset_paths.train`

## 说明

- authoritative `target_spec` 仍由 Reader 构造
- CLI full workflow 和 `run_multiagent.py` 都共享这层初始化逻辑
