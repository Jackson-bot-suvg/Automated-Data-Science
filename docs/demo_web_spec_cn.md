# 主项目 Demo Web 设计规范与功能文档

## 1. Demo 定位

这个网页是 CS33-2 主项目的实时演示指挥台，用来可视化展示本地数据分析任务如何被多 Agent 工作流执行。评审者打开页面后，第一眼应该看到任务、执行状态、Agent 编排树、日志和最终分析结果，而不是介绍型 landing page。

核心目标：

- 选择一个本地数据分析任务集。
- 点击开始后进入实时运行态。
- 右侧 50% 区域展示 Agent 编排树，并高亮当前执行 Agent。
- 日志窗口持续输出类似 test log 的进度文字。
- 结果区逐步展示分析指标、本地输出和 business report 链接。

## 2. 视觉规范

视觉方向采用 Industrial Data Ops / Mission Control 风格：深色工作台、精密网格、低饱和面板、cyan 运行信号、amber 等待状态、green 完成状态。整体要像一个真实的数据分析中控台，而不是普通后台模板。

设计原则：

- 第一屏就是可操作 demo。
- 不做营销 hero。
- 不使用大面积紫色渐变。
- 不使用装饰性漂浮圆点或无意义背景元素。
- 动效只服务于执行状态表达。
- Agent 编排树是视觉记忆点，要比普通流程图更有“正在计算”的感觉。

推荐色彩：

```css
--bg-main: #080d10;
--bg-panel: #10171b;
--bg-panel-2: #151f24;
--line-soft: #26343b;
--text-main: #e8eef0;
--text-muted: #8b9aa3;
--signal-cyan: #39d5cf;
--signal-amber: #f0b84b;
--signal-green: #7bd88f;
--signal-red: #ff6b5f;
--ink-blue: #4c83ff;
```

字体建议：

- 标题与数字：窄体技术感字体，例如 Rajdhani。
- 正文：高可读性 sans-serif。
- 日志和 ID：等宽字体。

## 3. 布局结构

桌面端采用固定主结构：

```text
+------------------------------------------------------------+
| 顶部状态栏：项目名 / 当前任务 / 运行状态 / 计时              |
+------------------------------+-----------------------------+
| 左侧 50%                     | 右侧 50%                   |
|                              |                             |
| 上半：任务集选择              | Agent 编排树                |
|                              | 当前节点高亮、连线流动       |
|------------------------------|                             |
| 下半：结果区                  | 下方小日志框                |
| 指标 / 图表 / report 链接     | Agent 进度文字              |
+------------------------------+-----------------------------+
```

左侧 50%：

- 上方：任务集选择和启动按钮。
- 下方：分析结果、指标、图表和报告链接。

右侧 50%：

- 主区域：Agent 编排树。
- 底部：实时进度日志。

## 4. 任务集选择功能

每个任务卡片展示：

- 任务名称。
- 数据集路径。
- 数据规模。
- 目标列或目标范围。
- 预计运行时间。
- 预期输出。

当前 demo 内置 8 个 `data/` 目录任务：

- Bank Churn Prediction。
- Academic Success Classification。
- Creature Type Classification。
- House Price Regression。
- Obesity Risk Classification。
- Steel Plate Defects。
- Spaceship Titanic。
- Titanic Survival。

交互规则：

- 默认选中 Bank Churn Prediction。
- 切换任务后，右侧任务详情更新。
- 运行中不允许切换任务，避免演示状态混乱。
- 点击 Start Analysis 后重置结果区和日志，并开始模拟事件流。

## 5. Agent 编排树功能

编排树映射仓库中的 LangGraph 节点：

```text
Reader
  -> Background Reviewer
  -> Planner
  -> Preliminary EDA
  -> Reviewer -> Planner
  -> Data Cleaning
  -> Reviewer -> Planner
  -> In-depth EDA
  -> Reviewer -> Planner
  -> Feature Engineering
  -> Reviewer -> Planner
  -> Model Building
  -> Reviewer
  -> Summarizer
  -> Business Translator
```

节点状态：

- idle：未执行，低亮度。
- queued：排队，amber 边框。
- running：当前执行，cyan 高亮、呼吸光环、扫描线。
- success：完成，green 状态。
- error：失败，red 状态，后续可接真实错误事件。

运行态视觉：

- 当前节点有 cyan 光环。
- 当前路径连线有流动虚线。
- 节点内部显示进度百分比。
- 顶部 chip 展示当前正在运行的 Agent。

## 6. 日志功能

日志位于右侧底部，形式接近测试日志：

```text
[00:02] Reader started: dataset intake.
[00:03] Reader: Resolved train/test/submission contract.
[00:04] Reader completed: Dataset contract accepted.
```

支持：

- 自动滚动。
- 暂停自动滚动。
- 清空日志。
- running 日志使用 cyan 左边线。
- success 日志使用 green 左边线。

## 7. 结果区功能

结果区位于左下方，随执行进度逐步更新。

指标：

- Rows analyzed。
- Data quality。
- Validation score。
- Business signal。

图表：

- 当前版本使用轻量内置柱状图表现分析流推进。
- 后续可替换为真实 EDA 图、模型对比图、业务指标图。

报告链接：

- `../artifacts/reports/business_report.md`
- `../artifacts/reports/technical_report.md`
- `../artifacts/reports/run_manifest.json`

报告未完成前链接禁用；Business Translator 完成后解锁。

## 8. 后续接真实后端的事件协议

当前网页已接入真实主项目 workflow：

- `python demo_web/server.py --port 8765` 启动 demo server。
- 前端点击 `Start Analysis` 会调用 `POST /api/demo-runs`。
- server 会启动 `main.py --yes --demo-log-file ...` 执行对应数据集任务。
- 前端通过 `GET /api/demo-runs/{runId}/events` 的 SSE 流读取真实 `DemoLogger` JSONL 事件。
- Agent tree 的节点高亮来自真实 `node_start` / `node_result`。
- `Time used` 来自真实节点 `latency_ms` / workflow `latency_ms`。
- `Tokens used` 来自真实 `token_usage_by_module.total_tokens`。

事件结构：

```ts
type DemoRunEvent =
  | { type: "run_started"; taskSetId: string }
  | { type: "agent_started"; agentId: string; message: string }
  | { type: "agent_progress"; agentId: string; progress: number; message: string }
  | { type: "agent_completed"; agentId: string; message: string }
  | { type: "metric_updated"; metricId: string; value: string; trend?: string }
  | { type: "report_ready"; markdownUrl: string; htmlUrl?: string }
  | { type: "run_completed"; durationSeconds: number }
  | { type: "run_failed"; error: string };
```

## 9. 验收标准

- 桌面端左右区域严格各占 50%。
- 左侧上方是任务区，左侧下方是结果区。
- 右侧主视觉是 Agent 编排树。
- Agent 节点执行时有明显 fancy processing 状态。
- 日志与 Agent 高亮同步推进。
- 指标和 report 链接随运行进度出现。
- 页面可作为静态文件直接演示，也可通过本地 HTTP server 打开。
