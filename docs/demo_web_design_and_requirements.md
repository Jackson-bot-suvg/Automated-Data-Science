# Demo Web Design and Functional Requirements

## 1. Product Intent

This demo web interface is a real-time command surface for the CS33-2 multi-agent data analysis workflow. It should let a reviewer immediately understand three things:

- A local analysis task set has been selected.
- A multi-agent workflow is executing.
- The system is producing measurable analysis results and a business report.

The page is not a landing page. The first screen is the operational demo itself.

## 2. Visual Direction

The interface uses an industrial data-operations aesthetic: dark operational panels, precise grid lines, restrained signal colors, and a high-energy animated agent orchestration tree. The memorable visual moment should be the right-side orchestration tree pulsing as work moves through agents.

Design principles:

- Dense but readable command-center layout.
- No generic SaaS hero section.
- No large purple gradients or decorative blobs.
- Use motion only where it explains task execution.
- Keep controls predictable and demo-safe.

## 3. Required Layout

The viewport is split into two equal columns.

```text
+------------------------------------------------------------+
| Top status bar: project, selected task, run status, timer   |
+------------------------------+-----------------------------+
| Left 50%                     | Right 50%                   |
|                              |                             |
| Top: task selection          | Agent orchestration tree     |
|                              | with live status animation   |
|------------------------------|                             |
| Bottom: results and metrics  | Bottom console progress log  |
|                              |                             |
+------------------------------+-----------------------------+
```

Left column:

- Top half: task set selection and launch controls.
- Bottom half: local analysis results, metrics, visual summaries, and business report links.

Right column:

- Main area: agent orchestration tree, occupying the visual priority of the right half.
- Lower area: compact progress console with text updates similar to test logs.

## 4. Visual System

Color tokens:

```css
--bg-main: #080d10;
--bg-panel: #10171b;
--bg-panel-2: #151f24;
--line-soft: #26343b;
--line-strong: #3d5159;
--text-main: #e8eef0;
--text-muted: #8b9aa3;
--signal-cyan: #39d5cf;
--signal-amber: #f0b84b;
--signal-green: #7bd88f;
--signal-red: #ff6b5f;
--ink-blue: #4c83ff;
```

Typography:

- Display and metric numbers: narrow technical display font.
- Body: highly readable sans-serif.
- Logs and IDs: monospace.

Panel rules:

- 6px to 8px radius.
- 1px borders.
- Low shadow usage.
- Current execution states may glow; idle states should remain quiet.

## 5. Task Set Selection

Task cards should include:

- Task name.
- Dataset path.
- Analysis objective.
- Dataset shape.
- Target column or target scope.
- Estimated runtime.
- Expected outputs.

Initial demo task sets:

- Bank churn prediction.
- Academic success classification.
- Creature type classification.
- House price regression.
- Obesity risk classification.
- Steel plate defect multi-label classification.
- Spaceship Titanic classification.
- Titanic survival classification.

User flow:

1. The first task is selected by default.
2. User can switch task cards.
3. The selected task detail panel updates.
4. User clicks Start Analysis.
5. The run state resets and the event stream begins.

## 6. Agent Orchestration Tree

The tree mirrors the repository workflow:

```text
Reader Agent
  -> Background Reviewer
  -> Planner Agent
  -> Preliminary EDA
  -> Reviewer Agent -> Planner Agent
  -> Data Cleaning
  -> Reviewer Agent -> Planner Agent
  -> In-depth EDA
  -> Reviewer Agent -> Planner Agent
  -> Feature Engineering
  -> Reviewer Agent -> Planner Agent
  -> Model Building
  -> Reviewer Agent
  -> Summarizer Agent
  -> Business Translator
```

Each agent has:

- ID.
- Display name.
- Role.
- Status: idle, queued, running, success, error.
- Progress percentage.
- Current message.

Running state behavior:

- Cyan ring pulse around the active node.
- Animated scanner line inside the node.
- Moving light on the active edge.
- Progress percentage visible inside the node.

Completed state behavior:

- Green status marker.
- Progress fixed at 100%.
- Node remains visible but calms down.

Queued state behavior:

- Amber marker.
- Muted node body.

## 7. Progress Console

The console presents structured live events in a compact log format.

Example:

```text
[00:04] Reader Agent inspected data/bank_churn and found train/test/submission files.
[00:11] Data Cleaning resolved 42 missing values and 18 duplicate-like records.
[00:22] Model Building evaluated LightGBM, XGBoost, and RandomForest candidates.
[00:34] Business Translator exported business_report.md.
```

Console controls:

- Auto-scroll toggle.
- Clear log button.
- Active-agent events are highlighted.
- Error lines are visually distinct.

## 8. Results Area

Results should appear progressively as the event stream advances.

Sections:

- Run summary.
- Key metrics.
- Visual analysis strips.
- Business report links.

Required metrics:

- Rows analyzed.
- Data quality score.
- Best model.
- Validation score.
- Key business insight.
- Report status.

Report links:

- Business report markdown.
- Technical report markdown.
- Run manifest.

Initial local paths:

- `../artifacts/reports/business_report.md`
- `../artifacts/reports/technical_report.md`
- `../artifacts/reports/run_manifest.json`

## 9. Event Model

The demo now uses the real project workflow. Run `python demo_web/server.py --port 8765` from the repository root. The frontend starts a real `main.py --yes --demo-log-file ...` subprocess through `POST /api/demo-runs`, then subscribes to real DemoLogger JSONL events through `GET /api/demo-runs/{runId}/events`.

The agent tree is driven by real `node_start` and `node_result` events. `Time used` is derived from real node/workflow latency. `Tokens used` is derived from `token_usage_by_module.total_tokens`.

```ts
type DemoRunEvent =
  | { type: "run_started"; taskSetId: string }
  | { type: "agent_started"; agentId: string; message: string }
  | { type: "agent_progress"; agentId: string; progress: number; message: string }
  | { type: "agent_completed"; agentId: string; message: string }
  | { type: "metric_updated"; metricId: string; value: string; trend?: string }
  | { type: "report_ready"; htmlUrl?: string; markdownUrl: string }
  | { type: "run_completed"; durationSeconds: number }
  | { type: "run_failed"; error: string };
```

## 10. Demo Acceptance Criteria

- The layout is exactly split: left 50%, right 50% on desktop.
- The left column has task selection above and results below.
- The right column clearly prioritizes the orchestration tree.
- Selecting a task updates visible task details.
- Starting a run animates the agent tree in sequence.
- Logs update in real time.
- Metrics and report links become available during/after execution.
- The page works as a standalone local web page.
