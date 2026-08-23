# 日志与可观测性指南（含 LangSmith 改动说明）

本文档汇总日志与 LangSmith 的改动范围、关键行为与使用方法。

## 1. 改动概览

本轮主要完成以下接入：

- 在 `main.py`、`run_multiagent.py`、`tests/test_pipeline_no_model.py` 中接入本地结构化日志。
- 接入 LangSmith tracing（workflow/phase 维度）。
- 本地日志自动命名规则：`<数据集>_<时间>.jsonl`，避免覆盖历史日志。
  - `main.py` 默认写入到 `artifacts/runs/<run_label>/logs/`（每次运行一个目录）
  - 可视化产物默认写入到 `artifacts/runs/<run_label>/figures/`
- 将 token usage 聚合到本地日志，并在可用时同步到 LangSmith metadata。
- 增加日志可视化（节点耗时、waterfall、token 消耗图、token I/O + prompt 图）。
- `main.py` 内置可视化生成，运行结束自动产图。

## 2. 关键改动文件

- `main.py`
  - 本地日志写入（`DemoLogger`）
  - LangSmith tracing context（Reader/Reviewer/full workflow）
  - 自动生成可视化图表
- `run_multiagent.py`
  - 按数据集+时间自动命名日志
  - 记录 workflow/node 事件与错误
- `tests/test_pipeline_no_model.py`
  - 测试流程本地日志与 tracing 接入
- `app/llm/providers/openai_compat.py`
  - tracing 开启时使用 `wrap_openai`，便于 LangSmith 识别 LLM usage
- `app/llm/service.py`
  - 支持 `module_name`，记录 usage + 输入输出到本地聚合器
- `app/llm/helpers.py`、`app/planners/base.py`
  - 调用 LLM 时传入 `module_name`
- `app/observability/token_usage.py`
  - 聚合模块 token 使用量，并保留最近交互 input/output（截断）
- `app/observability/demo_logger.py`
  - 节点日志增加 `token_usage` 与 `token_usage_by_module`
- `monitor_demo_observability.py`
  - LangSmith 回拉结果可落盘本地 JSONL
- `visualize_json_logs.py`
  - 生成节点耗时图、waterfall、token 使用图、token I/O + prompt 图

## 3. 日志来源与文件命名

### 3.1 本地 workflow 日志

- 路径：`artifacts/runs/<run_label>/logs/`（`main.py` 默认）
- 默认命名：`<dataset>_<YYYYMMDD_HHMMSS>.jsonl`（同名 stem 用于 root trace）
- 示例：`artifacts/runs/bank_churn_20260429_175517_96d549dd/logs/bank_churn_20260429_175517.jsonl`

如你在命令行显式传了 `--demo-log-file`，则会写入你指定的位置（用于兼容旧流程/手动回放）。

常见事件类型：

- `workflow_start`
- `node_start`
- `node_result`（`done` / `error`）
- `workflow_end`
- `workflow_error`

### 3.2 LangSmith 回拉日志

- 默认路径：`artifacts/logs/<dataset>_<YYYYMMDD_HHMMSS>_langsmith.jsonl`
- 由 `monitor_demo_observability.py` 生成

说明：`monitor_demo_observability.py` 会自动回找本地 demo JSONL（现在支持 `artifacts/runs/*/logs/`）。若你希望 LangSmith 回拉结果也落到 run 目录，可用 `--ls-log-file` 指定输出文件。

常见事件类型：

- `langsmith_run_update`
- `langsmith_poll_error`

## 4. 关键字段说明

节点完成事件（`node_result`）常见字段：

- `node_name`, `status`, `latency_ms`
- `review_status`（通常只在 reviewer 类节点有值）
- `token_usage`（当前节点映射到的模块 usage）
- `token_usage_by_module`（模块级全量快照）

`token_usage_by_module` 中包含：

- `llm_calls`, `input_tokens`, `output_tokens`, `total_tokens`
- `recent_interactions`
  - `input`（消息列表，截断）
  - `output`（文本，截断）
  - `model`, `input_tokens`, `output_tokens`, `total_tokens`

LangSmith 回拉日志关键字段：

- `id`, `run_name`, `run_type`, `status`
- `thread_id`, `start_time`, `end_time`, `latency_ms`
- `input_tokens`, `output_tokens`, `total_tokens`, `total_cost`（可用时）
- `error`, `polled_at`

## 5. 实时监控建议流程

推荐 demo 时使用两个终端：

1) 终端 A：运行 workflow

```bash
python run_multiagent.py titanic
```

2) 终端 B：同时监控本地与 LangSmith 状态

```bash
python monitor_demo_observability.py --project 26s1-cs33-2-dev --thread-id titanic-cli-run --dataset titanic
```

终端 B 输出中：

- `[LOCAL] ...` 表示本地工作流事件
- `[LS] ...` 表示 LangSmith 回拉事件

## 6. LangSmith 配置要求

`.env` 需包含：

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=<your_key>`
- `LANGSMITH_PROJECT=<your_project>`

可选：

- `LANGSMITH_WORKSPACE_ID=<workspace_id>`
- `APP_ENV=dev|test|prod`

建议同时确认：

- 依赖已安装 `langsmith`
- OpenAI-compatible provider 走 `wrap_openai` 路径

## 7. 常用命令

### 7.1 main（自动日志 + 自动可视化）

```bash
python main.py --task "predict customer churn" --dataset data/bank_churn --yes
```

运行结束后，图表一般在 `artifacts/runs/<run_label>/figures/` 下生成。

### 7.2 run_multiagent

```bash
python run_multiagent.py titanic
```

### 7.3 监控本地 + LangSmith 回拉

```bash
python monitor_demo_observability.py --project 26s1-cs33-2-dev --thread-id titanic-cli-run --dataset titanic
```

### 7.4 手动可视化（可选）

```bash
python visualize_json_logs.py --input artifacts/runs/<run_label>/logs/<your_log>.jsonl --output-dir artifacts/runs/<run_label>/figures --scope latest
```

## 8. 常见问题排查

### Q1: `review_status` 为什么为空？

多数节点不产出 reviewer 状态；只有 reviewer 类节点通常有值，属于预期。

### Q2: token 为什么仍为空？

常见原因：

- 使用的是改造前旧日志
- 本次节点未触发 LLM 调用
- 节点名与模块名不一致，节点级 `token_usage` 可能为空（优先看 `token_usage_by_module`）

### Q3: LangSmith 看不到数据？

排查顺序：

1. `.env` 中 `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`
2. provider 是否启用 `wrap_openai`
3. 网络/代理是否阻断 LangSmith API

### Q4: Dashboard 有 trace 但 token/cost 为 0？

- 调用链可能已上报，但 usage 尚未被识别或聚合延迟
- 在 trace 详情页检查 LLM run 的 usage 字段

## 9. 建议实践

- 统计时优先使用 `token_usage_by_module`，`token_usage` 仅作节点参考。
- Demo 建议使用“双通道”：本地 JSONL 实时展示 + LangSmith 云端复盘。
- 建议将 `main.py` 的日志统一保存在 `artifacts/runs/<run_label>/logs/`（每次运行独立目录，便于定位与回放）。

## 10. 最新 LangSmith Trace 合并改动（2026-04）

本节记录最近针对“一个流程出现多个顶层 trace”的修复。

### 10.1 目标行为

- 一次 `main.py` 运行只产生一个主 trace（root trace）。
- Reader / Background Reviewer / LangGraph workflow / business_translator 的 LLM 调用都挂在同一棵 trace 树下。
- root trace 名称与本地日志一致，格式为 `<dataset>_<YYYYMMDD_HHMMSS>`。

### 10.2 关键实现

- `main.py`
  - 使用 `run_with_root_trace(...)` 包裹从确认阶段到完整 workflow 的执行链路。
  - `run_name` 直接使用 `demo_logger.path.stem`，确保 trace 名称与日志文件名一一对应。
- `app/observability/langsmith.py`
  - 新增 `run_with_root_trace(...)`：基于 `@traceable(run_type="chain")` 显式创建 root trace。
  - 保留 `get_current_run_url()`、`get_latest_run_url()` 与 public URL 兜底获取能力。
- `app/planners/base.py`
  - 去掉 planner 内部二次创建 tracing context，避免拆分顶层 trace。
  - 改为通过 `trace_extra` 透传 tags/metadata。
- `app/agents/business_translator.py`
  - 修复线程桥接导致的 tracing 上下文丢失（`contextvars.copy_context()`）。
  - 避免 business_translator 内多次 LLM 调用被识别为 4 个独立顶层 trace。

### 10.3 URL 获取顺序（运行结束后打印）

- 优先：`get_current_public_run_url()`（直接获取当前 root trace 的 public URL）。
- 兜底：`get_latest_public_run_url(thread_id, dataset)`（按 metadata 回查最近 run 并回溯 root）。
- 再兜底：`get_current_run_url()` / `get_latest_run_url(...)`（非 public 的普通 trace URL）。

输出示例：

- `LangSmith public trace: https://smith.langchain.com/public/<token>/r`
- `LangSmith trace: https://smith.langchain.com/o/<workspace>/projects/p/<project>/r/<run_id>`

### 10.4 验收检查

完成一次运行后，建议检查：

1. LangSmith 列表页出现 1 条主 trace（名称为 `<dataset>_<timestamp>`）。
2. 展开主 trace 后可看到 reader/reviewer/planner/business_translator 等子调用。
3. 终端打印出 public trace URL（可直接分享复盘）。
