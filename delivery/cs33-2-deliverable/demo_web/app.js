if (window.location.protocol === "file:") {
  window.location.replace("http://127.0.0.1:8765/");
}

const taskSets = [
  {
    id: "bank-churn",
    name: "Bank Churn Prediction",
    dataset: "data/bank_churn",
    shape: "165,034 train rows",
    target: "Exited",
    modelType: "Binary classification",
    objective: "Predict customer churn and translate model drivers into retention actions.",
    rows: "165,034",
    quality: "96.8%",
    score: "0.872 AUC",
    signal: "High-value churn",
    insight:
      "Tenure, active membership, and balance concentration are the strongest signals for churn risk. The business report recommends retention outreach for high-balance inactive customers."
  },
  {
    id: "academic-success",
    name: "Academic Success Classification",
    dataset: "data/classification_with_an_academic_success_dataset",
    shape: "76,518 train rows",
    target: "Target",
    modelType: "Multiclass classification",
    objective: "Classify student academic outcomes and summarize intervention opportunities.",
    rows: "76,518",
    quality: "95.2%",
    score: "0.834 accuracy",
    signal: "At-risk cohort",
    insight:
      "Prior academic performance and enrollment context drive most outcome separation. The business report frames this as a retention and support prioritization problem."
  },
  {
    id: "ghouls",
    name: "Creature Type Classification",
    dataset: "data/ghouls_goblins_and_ghosts_boo",
    shape: "371 train rows",
    target: "type",
    modelType: "Multiclass classification",
    objective: "Classify compact tabular records with categorical and numeric morphology signals.",
    rows: "371",
    quality: "98.9%",
    score: "0.741 accuracy",
    signal: "Color interaction",
    insight:
      "Small-data behavior makes validation variance visible. The final report emphasizes confidence limits and which features remain stable across model candidates."
  },
  {
    id: "house-prices",
    name: "House Price Regression",
    dataset: "data/house_prices",
    shape: "1,460 train rows",
    target: "SalePrice",
    modelType: "Regression",
    objective: "Estimate property sale price and surface the highest-value property attributes.",
    rows: "1,460",
    quality: "94.1%",
    score: "0.132 RMSLE",
    signal: "Quality premium",
    insight:
      "Overall quality, living area, garage capacity, and neighborhood dominate price movement. Report guidance focuses on valuation confidence and renovation ROI."
  },
  {
    id: "obesity-risks",
    name: "Obesity Risk Classification",
    dataset: "data/obesity_risks",
    shape: "20,758 train rows",
    target: "NObeyesdad",
    modelType: "Multiclass classification",
    objective: "Classify obesity risk levels and translate health-pattern drivers into segment guidance.",
    rows: "20,758",
    quality: "96.1%",
    score: "0.904 accuracy",
    signal: "Lifestyle segment",
    insight:
      "Physical activity, eating frequency, and family-history features create clear segment boundaries. The report focuses on targeted prevention messaging."
  },
  {
    id: "plate-defect",
    name: "Steel Plate Defects",
    dataset: "data/plate_defect",
    shape: "multi-label train/test",
    target: "7 defect labels",
    modelType: "Multilabel classification",
    objective: "Detect plate defect classes and summarize operational quality risks.",
    rows: "19,219",
    quality: "97.4%",
    score: "0.889 macro AUC",
    signal: "Surface fault mix",
    insight:
      "Scratch and stain classes separate cleanly from geometric features. The business report flags inspection priority and likely process-control interventions."
  },
  {
    id: "spaceship",
    name: "Spaceship Titanic",
    dataset: "data/spaceship_titanic",
    shape: "8,693 train rows",
    target: "Transported",
    modelType: "Binary classification",
    objective: "Classify passenger transport outcome and explain cohort-level movement.",
    rows: "8,693",
    quality: "95.6%",
    score: "0.812 accuracy",
    signal: "CryoSleep cohort",
    insight:
      "CryoSleep status and destination cohorts dominate the decision boundary. The final report frames this as cohort segmentation and operational likelihood."
  },
  {
    id: "titanic",
    name: "Titanic Survival",
    dataset: "data/titanic_autokaggle",
    shape: "891 train rows",
    target: "Survived",
    modelType: "Binary classification",
    objective: "Predict passenger survival and produce an interpretable benchmark analysis.",
    rows: "891",
    quality: "97.0%",
    score: "0.821 accuracy",
    signal: "Class and sex split",
    insight:
      "Passenger class, sex, fare, and family structure remain the most stable drivers. The report is useful as a compact end-to-end workflow benchmark."
  }
];

const agents = [
  { id: "reader", label: "Reader", role: "dataset intake", x: 17, y: 10 },
  { id: "background_reviewer", label: "Background Reviewer", role: "task validation", x: 17, y: 25 },
  { id: "planner", label: "Planner", role: "phase routing", x: 17, y: 88 },
  { id: "preliminary_eda", label: "Preliminary EDA", role: "first pass stats", x: 52, y: 10 },
  { id: "data_cleaning", label: "Data Cleaning", role: "data repair", x: 52, y: 30 },
  { id: "indepth_eda", label: "In-depth EDA", role: "signal search", x: 52, y: 50 },
  { id: "feature_engineering", label: "Feature Engineering", role: "feature build", x: 52, y: 70 },
  { id: "model_building", label: "Model Building", role: "candidate models", x: 52, y: 90 },
  { id: "reviewer", label: "Reviewer", role: "phase gate", x: 84, y: 28 },
  { id: "summarizer", label: "Summary", role: "technical report", x: 84, y: 52 },
  { id: "business_translator", label: "Business Translator", role: "business report", x: 84, y: 76 }
];

const edges = [
  { from: "reader", to: "background_reviewer", route: "vertical" },
  { from: "background_reviewer", to: "planner", route: "vertical" },
  { from: "planner", to: "preliminary_eda", route: "fan" },
  { from: "preliminary_eda", to: "reviewer", route: "direct" },
  { from: "reviewer", to: "planner", route: "return" },
  { from: "planner", to: "data_cleaning", route: "fan" },
  { from: "data_cleaning", to: "reviewer", route: "direct" },
  { from: "planner", to: "indepth_eda", route: "fan" },
  { from: "indepth_eda", to: "reviewer", route: "direct" },
  { from: "planner", to: "feature_engineering", route: "fan" },
  { from: "feature_engineering", to: "reviewer", route: "direct" },
  { from: "planner", to: "model_building", route: "fan" },
  { from: "model_building", to: "reviewer", route: "direct" },
  { from: "reviewer", to: "summarizer", route: "vertical" },
  { from: "summarizer", to: "business_translator", route: "vertical" }
];

const agentMessages = {
  reader: ["Inspecting dataset folder", "Resolved train/test/submission contract", "Dataset contract accepted"],
  background_reviewer: ["Checking task feasibility", "Validating inferred target scope", "Task understanding approved"],
  planner: ["Building phase plan", "Selecting next graph node", "Phase route locked"],
  preliminary_eda: ["Profiling columns and target", "Generating first-pass distributions", "EDA baseline ready"],
  data_cleaning: ["Repairing missing values", "Checking duplicate-like records", "Clean train/test artifacts ready"],
  indepth_eda: ["Searching drivers and anomalies", "Building target relationship views", "Insight candidates ready"],
  feature_engineering: ["Encoding categorical signals", "Building derived feature set", "Feature summary exported"],
  model_building: ["Training candidate models", "Comparing validation metrics", "Best model selected"],
  reviewer: ["Auditing outputs", "Checking phase evidence", "Quality gate passed"],
  summarizer: ["Compiling technical report", "Linking metrics and artifacts", "Technical report exported"],
  business_translator: ["Translating findings", "Writing business recommendations", "Business report exported"]
};

const state = {
  selectedTaskId: taskSets[0].id,
  running: false,
  runId: null,
  startedAt: null,
  timerId: null,
  eventTimers: [],
  eventSource: null,
  autoScroll: true,
  activeAgentId: null,
  activeEdgeKey: null,
  lastAgentId: null,
  completedAgents: new Set(),
  totalLatencyMs: 0,
  totalTokens: 0,
  progressByAgent: Object.fromEntries(agents.map((agent) => [agent.id, 0])),
  snapshotRequestId: 0,
  reportPaths: {
    business: "",
    technical: "",
    manifest: ""
  }
};

const elements = {
  activeTaskName: document.getElementById("activeTaskName"),
  runStatus: document.getElementById("runStatus"),
  elapsedTime: document.getElementById("elapsedTime"),
  taskList: document.getElementById("taskList"),
  taskDetail: document.getElementById("taskDetail"),
  resetButton: document.getElementById("resetButton"),
  reportPill: document.getElementById("reportPill"),
  metricRows: document.getElementById("metricRows"),
  metricRowsTrend: document.getElementById("metricRowsTrend"),
  metricQuality: document.getElementById("metricQuality"),
  metricQualityTrend: document.getElementById("metricQualityTrend"),
  metricScore: document.getElementById("metricScore"),
  metricScoreTrend: document.getElementById("metricScoreTrend"),
  metricSignal: document.getElementById("metricSignal"),
  metricSignalTrend: document.getElementById("metricSignalTrend"),
  chartStatus: document.getElementById("chartStatus"),
  miniChart: document.getElementById("miniChart"),
  insightBox: document.getElementById("insightBox"),
  reportLinks: document.getElementById("reportLinks"),
  activeAgentChip: document.getElementById("activeAgentChip"),
  edgeLayer: document.getElementById("edgeLayer"),
  agentLayer: document.getElementById("agentLayer"),
  logFeed: document.getElementById("logFeed"),
  scrollButton: document.getElementById("scrollButton"),
  clearLogButton: document.getElementById("clearLogButton")
};

function selectedTask() {
  return taskSets.find((task) => task.id === state.selectedTaskId) || taskSets[0];
}

function formatTime(totalSeconds) {
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function renderTasks() {
  elements.taskList.innerHTML = taskSets
    .map(
      (task) => `
        <button class="task-card ${task.id === state.selectedTaskId ? "active" : ""}" data-task-id="${task.id}" type="button">
          <strong>${task.name}</strong>
          <small>${task.dataset}</small>
        </button>
      `
    )
    .join("");

  elements.taskList.querySelectorAll(".task-card").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.running) return;
      state.selectedTaskId = button.dataset.taskId;
      renderTasks();
      resetRun({ keepSelection: true });
      renderTaskDetail();
      updateHeader();
    });
  });
}

function renderTaskDetail() {
  const task = selectedTask();
  elements.taskDetail.innerHTML = `
    <h3>${task.name}</h3>
    <p>${task.objective}</p>
    <div class="detail-grid">
      <div class="detail-cell">
        <span>Dataset</span>
        <strong>${task.dataset}</strong>
      </div>
      <div class="detail-cell">
        <span>Target</span>
        <strong>${task.target}</strong>
      </div>
      <div class="detail-cell">
        <span>Shape</span>
        <strong>${task.shape}</strong>
      </div>
      <div class="detail-cell">
        <span>Modeling type</span>
        <strong>${task.modelType}</strong>
      </div>
    </div>
    <button class="start-button" id="startButton" type="button">${state.running ? "Running Analysis" : "Start Analysis"}</button>
  `;

  document.getElementById("startButton").addEventListener("click", startRun);
  document.getElementById("startButton").classList.toggle("running", state.running);
}

function updateHeader() {
  const task = selectedTask();
  elements.activeTaskName.textContent = task.name;
  elements.runStatus.textContent = state.running ? "Running" : state.completedAgents.size ? "Complete" : "Ready";
}

function renderTree() {
  elements.edgeLayer.innerHTML = "";
  elements.agentLayer.innerHTML = agents.map(renderAgentNode).join("");
  renderMeasuredEdges();
  requestAnimationFrame(renderMeasuredEdges);
}

function renderMeasuredEdges() {
  const stage = document.getElementById("treeStage");
  const stageRect = stage.getBoundingClientRect();
  elements.edgeLayer.setAttribute("viewBox", `0 0 ${stageRect.width} ${stageRect.height}`);
  elements.edgeLayer.innerHTML = edges.map((edge) => renderMeasuredEdge(edge, stageRect)).join("");
}

function renderMeasuredEdge(edge, stageRect) {
  const fromNode = elements.agentLayer.querySelector(`[data-agent-id="${edge.from}"]`);
  const toNode = elements.agentLayer.querySelector(`[data-agent-id="${edge.to}"]`);
  if (!fromNode || !toNode) return "";

  const from = rectRelativeToStage(fromNode.getBoundingClientRect(), stageRect);
  const to = rectRelativeToStage(toNode.getBoundingClientRect(), stageRect);
  const active = state.activeEdgeKey === `${edge.from}->${edge.to}`;
  const done = state.completedAgents.has(edge.from) && state.completedAgents.has(edge.to);
  const cls = `edge ${edge.route === "return" ? "return" : ""} ${active ? "active" : ""} ${done ? "done" : ""}`;
  return `<path class="${cls}" d="${edgePath(edge.route, from, to, stageRect)}" />`;
}

function rectRelativeToStage(rect, stageRect) {
  return {
    left: rect.left - stageRect.left,
    right: rect.right - stageRect.left,
    top: rect.top - stageRect.top,
    bottom: rect.bottom - stageRect.top,
    centerX: rect.left - stageRect.left + rect.width / 2,
    centerY: rect.top - stageRect.top + rect.height / 2
  };
}

function edgePath(route, from, to, stageRect) {
  if (route === "vertical") {
    const down = to.centerY >= from.centerY;
    const start = { x: from.centerX, y: down ? from.bottom : from.top };
    const end = { x: to.centerX, y: down ? to.top : to.bottom };
    const midY = (start.y + end.y) / 2;
    return `M ${start.x} ${start.y} C ${start.x} ${midY}, ${end.x} ${midY}, ${end.x} ${end.y}`;
  }

  if (route === "return") {
    const start = { x: from.left, y: from.bottom - 8 };
    const end = { x: to.right, y: to.bottom - 8 };
    const busY = Math.min(stageRect.height - 34, Math.max(start.y, end.y) + 76);
    return `M ${start.x} ${start.y} L ${start.x} ${busY} L ${end.x} ${busY} L ${end.x} ${end.y}`;
  }

  const start = { x: from.right, y: from.centerY };
  const end = { x: to.left, y: to.centerY };
  const delta = Math.max(34, Math.abs(end.x - start.x) * (route === "fan" ? 0.36 : 0.5));
  return `M ${start.x} ${start.y} C ${start.x + delta} ${start.y}, ${end.x - delta} ${end.y}, ${end.x} ${end.y}`;
}

function renderAgentNode(agent) {
  const status = getAgentStatus(agent.id);
  const progress = state.progressByAgent[agent.id] || 0;
  return `
    <div class="agent-node ${status}" style="left:${agent.x}%; top:${agent.y}%;" data-agent-id="${agent.id}">
      <div class="agent-dot">${status === "success" ? "OK" : String(progress).padStart(2, "0")}</div>
      <div class="agent-copy">
        <strong>${agent.label}</strong>
        <span>${agent.role}</span>
      </div>
      <div class="agent-progress">${progress}%</div>
    </div>
  `;
}

function getAgentStatus(agentId) {
  if (state.activeAgentId === agentId) return "running";
  if (state.completedAgents.has(agentId)) return "success";
  const activeIndex = agents.findIndex((agent) => agent.id === state.activeAgentId);
  const index = agents.findIndex((agent) => agent.id === agentId);
  if (state.running && activeIndex >= 0 && index > activeIndex) return "queued";
  return "idle";
}

function logLine(message, mode = "active") {
  const elapsed = state.startedAt ? Math.floor((Date.now() - state.startedAt) / 1000) : 0;
  const row = document.createElement("div");
  row.className = `log-line ${mode}`;
  row.innerHTML = `<span class="log-time">[${formatTime(elapsed)}]</span><span>${message}</span>`;
  elements.logFeed.appendChild(row);
  elements.logFeed.scrollTop = elements.logFeed.scrollHeight;
}

function formatMetricValue(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(4);
  }
  return value ? String(value) : "--";
}

function metricNameLabel(metricName) {
  if (!metricName) return "validation metric";
  return String(metricName);
}

function syncReportLinks() {
  elements.reportLinks.querySelectorAll("[data-report-kind]").forEach((link) => {
    const kind = link.dataset.reportKind;
    const href = state.reportPaths[kind];
    if (href) {
      link.href = href;
      link.classList.remove("disabled");
    } else {
      link.href = "#";
      link.classList.add("disabled");
    }
  });
}

function resetRun(options = {}) {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.eventTimers.forEach((timer) => clearTimeout(timer));
  state.eventTimers = [];
  clearInterval(state.timerId);
  state.running = false;
  state.runId = null;
  state.startedAt = null;
  state.activeAgentId = null;
  state.activeEdgeKey = null;
  state.lastAgentId = null;
  state.completedAgents = new Set();
  state.totalLatencyMs = 0;
  state.totalTokens = 0;
  state.progressByAgent = Object.fromEntries(agents.map((agent) => [agent.id, 0]));
  state.reportPaths = { business: "", technical: "", manifest: "" };
  elements.elapsedTime.textContent = "00:00";
  elements.logFeed.innerHTML = "";
  elements.reportPill.textContent = "Report pending";
  elements.reportPill.classList.remove("ready");
  elements.metricRows.textContent = "--";
  elements.metricRowsTrend.textContent = "Waiting for Reader";
  elements.metricQuality.textContent = "--";
  elements.metricQualityTrend.textContent = "Waiting for workflow events";
  elements.metricScore.textContent = "--";
  elements.metricScoreTrend.textContent = "Waiting for Model";
  elements.metricSignal.textContent = "--";
  elements.metricSignalTrend.textContent = "Waiting for LLM usage";
  elements.chartStatus.textContent = "Standby";
  elements.miniChart.classList.remove("hot");
  elements.insightBox.textContent = "Select a task and start the run to populate local findings.";
  elements.activeAgentChip.textContent = "No active agent";
  elements.activeAgentChip.classList.remove("running");
  syncReportLinks();
  updateHeader();
  renderTaskDetail();
  renderTree();
  if (!state.running) {
    void loadLatestSnapshotForSelectedTask();
  }
  if (!options.silent) {
    logLine(options.keepSelection ? "Run state reset for selected task." : "Command deck reset.", "success");
  }
}

async function startRun() {
  if (state.running) return;
  resetRun({ silent: true, keepSelection: true });
  const task = selectedTask();
  state.running = true;
  state.startedAt = Date.now();
  updateHeader();
  renderTaskDetail();
  elements.runStatus.textContent = "Running";
  elements.chartStatus.textContent = "Streaming";
  elements.miniChart.classList.add("hot");
  elements.metricRows.textContent = task.rows;
  elements.metricRowsTrend.textContent = `Dataset selected: ${task.dataset}`;
  logLine(`Starting real workflow for ${task.dataset}.`, "active");
  state.timerId = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.startedAt) / 1000);
    elements.elapsedTime.textContent = formatTime(elapsed);
    updateMetric("quality", formatDuration(elapsed * 1000), "Wall-clock elapsed time", { quiet: true });
  }, 250);
  try {
    const response = await fetch("/api/demo-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ taskSetId: task.id })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `Failed to start run: ${response.status}`);
    }
    state.runId = payload.runId;
    logLine(`Backend run created: ${payload.runId}.`, "success");
    attachEventStream(payload.runId);
  } catch (error) {
    logLine(`Unable to start backend workflow: ${error.message}`, "error");
    state.running = false;
    clearInterval(state.timerId);
    elements.runStatus.textContent = "Error";
    renderTaskDetail();
  }
}

function attachEventStream(runId) {
  state.eventSource = new EventSource(`/api/demo-runs/${encodeURIComponent(runId)}/events`);
  state.eventSource.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleBackendMessage(message);
  };
  state.eventSource.onerror = () => {
    logLine("Event stream disconnected.", "error");
    if (state.running) {
      elements.runStatus.textContent = "Stream lost";
    }
  };
}

function beginAgent(agent, previousAgentId = null) {
  if (!agent) return;
  state.activeAgentId = agent.id;
  state.activeEdgeKey = previousAgentId && edgeExists(previousAgentId, agent.id) ? `${previousAgentId}->${agent.id}` : null;
  state.lastAgentId = agent.id;
  state.progressByAgent[agent.id] = 8;
  elements.activeAgentChip.textContent = `${agent.label} running`;
  elements.activeAgentChip.classList.add("running");
  logLine(`${agent.label} started: ${agent.role}.`, "active");
  renderTree();
}

function progressAgent(agent, progress, message) {
  if (!agent) return;
  state.activeAgentId = agent.id;
  state.progressByAgent[agent.id] = progress;
  logLine(`${agent.label}: ${message}.`, "active");
  renderTree();
}

function finishAgent(agent, message) {
  if (!agent) return;
  state.progressByAgent[agent.id] = 100;
  state.completedAgents.add(agent.id);
  logLine(`${agent.label} completed: ${message}.`, "success");
  renderTree();
}

function updateMetric(metric, value, trend, options = {}) {
  const slots = {
    rows: [elements.metricRows, elements.metricRowsTrend],
    quality: [elements.metricQuality, elements.metricQualityTrend],
    score: [elements.metricScore, elements.metricScoreTrend],
    signal: [elements.metricSignal, elements.metricSignalTrend]
  };
  const [valueEl, trendEl] = slots[metric];
  valueEl.textContent = value;
  trendEl.textContent = trend;
  if (!options.quiet) {
    logLine(`Metric updated: ${trend}.`, "success");
  }
}

function completeRun(task, options = {}) {
  state.running = false;
  state.activeAgentId = null;
  state.activeEdgeKey = null;
  clearInterval(state.timerId);
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  elements.runStatus.textContent = options.error ? "Error" : "Complete";
  elements.activeAgentChip.textContent = "Workflow complete";
  elements.activeAgentChip.classList.remove("running");
  elements.reportPill.textContent = options.error ? "Run failed" : "Report ready";
  elements.reportPill.classList.toggle("ready", !options.error);
  elements.chartStatus.textContent = options.error ? "Error" : "Complete";
  elements.insightBox.textContent = task.insight;
  syncReportLinks();
  logLine(options.error ? "Workflow failed." : "Workflow completed.", options.error ? "error" : "success");
  renderTaskDetail();
  renderTree();
}

function handleBackendMessage(message) {
  if (message.type === "run_attached") {
    logLine(`Streaming real workflow events from ${message.logPath}.`, "success");
    return;
  }
  if (message.type === "console" && message.message) {
    if (/Workflow failed|Traceback|Error/i.test(message.message)) {
      logLine(message.message, "error");
    }
    return;
  }
  if (message.type === "workflow_event") {
    handleWorkflowEvent(message.payload);
    return;
  }
  if (message.type === "run_snapshot") {
    applyRunSnapshot(message.payload);
    return;
  }
  if (message.type === "process_exit") {
    if (message.returnCode !== 0) {
      completeRun(selectedTask(), { error: true });
    }
    return;
  }
  if (message.type === "stream_closed" && state.running) {
    completeRun(selectedTask());
  }
}

function handleWorkflowEvent(payload) {
  const event = payload.event;
  const nodeId = normalizeNodeId(payload.node_name);
  if (event === "workflow_start") {
    elements.runStatus.textContent = "Running";
    logLine(`Workflow started. thread=${payload.thread_id || "local"}`, "success");
    return;
  }
  if (event === "node_start" && nodeId) {
    const agent = agentById(nodeId);
    beginAgent(agent, state.lastAgentId);
    return;
  }
  if (event === "node_result" && nodeId) {
    const agent = agentById(nodeId);
    const isError = payload.status === "error";
    if (isError) {
      logLine(`${agent.label} failed: ${payload.error || "unknown error"}`, "error");
      completeRun(selectedTask(), { error: true });
      return;
    }
    if (typeof payload.latency_ms === "number") {
      state.totalLatencyMs += payload.latency_ms;
      updateMetric("quality", formatDuration(state.totalLatencyMs), "Cumulative node runtime");
    }
    updateTokensFromUsage(payload.token_usage_by_module || payload.token_usage || {});
    if (payload.technical_report_path) {
      updateReportLink("technical", payload.technical_report_path);
    }
    if (payload.business_report_path) {
      updateReportLink("business", payload.business_report_path);
    }
    finishAgent(agent, `${agentMessages[nodeId]?.[2] || "Completed"} (${formatDuration(payload.latency_ms || 0)})`);
    return;
  }
  if (event === "workflow_end") {
    if (typeof payload.latency_ms === "number") {
      updateMetric("quality", formatDuration(payload.latency_ms), "Total workflow runtime");
    }
    updateTokensFromUsage(payload.token_usage_by_module || {});
    if (payload.report_path && /business_report/i.test(payload.report_path)) {
      updateReportLink("business", payload.report_path);
    }
    completeRun(selectedTask());
    return;
  }
  if (event === "workflow_error") {
    logLine(`Workflow error: ${payload.error || "unknown error"}`, "error");
    completeRun(selectedTask(), { error: true });
  }
}

function normalizeNodeId(nodeName) {
  const raw = String(nodeName || "").trim();
  const aliases = {
    reader: "reader",
    background_reviewer: "background_reviewer",
    planner: "planner",
    preliminary_eda: "preliminary_eda",
    prelim_eda: "preliminary_eda",
    data_cleaning: "data_cleaning",
    indepth_eda: "indepth_eda",
    in_depth_eda: "indepth_eda",
    feature_engineering: "feature_engineering",
    model_building: "model_building",
    modeling: "model_building",
    reviewer: "reviewer",
    summarizer: "summarizer",
    summary: "summarizer",
    business_translator: "business_translator"
  };
  return aliases[raw] || null;
}

function agentById(agentId) {
  return agents.find((agent) => agent.id === agentId);
}

function edgeExists(fromId, toId) {
  return edges.some((edge) => edge.from === fromId && edge.to === toId);
}

function updateTokensFromUsage(usage) {
  const total = sumTokens(usage);
  if (total <= 0) return;
  state.totalTokens = Math.max(state.totalTokens, total);
  updateMetric("signal", state.totalTokens.toLocaleString(), "Total LLM tokens used");
}

function applyRunSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== "object") return;
  if (Number.isFinite(snapshot.trainRows) && snapshot.trainRows > 0) {
    updateMetric("rows", Number(snapshot.trainRows).toLocaleString(), "Train rows from run manifest", { quiet: true });
  }
  if (typeof snapshot.totalRuntimeSec === "number" && snapshot.totalRuntimeSec > 0) {
    updateMetric("quality", formatDuration(snapshot.totalRuntimeSec * 1000), "Workflow runtime from manifest", { quiet: true });
  }
  const metric = snapshot.primaryMetric || {};
  if (metric.name) {
    const scoreValue = metric.formatted || formatMetricValue(metric.value);
    updateMetric("score", scoreValue, `Primary metric: ${metricNameLabel(metric.name)}`, { quiet: true });
  }
  const reportPaths = snapshot.reportPaths || {};
  if (reportPaths.business) {
    updateReportLink("business", reportPaths.business);
    elements.reportPill.textContent = "Report ready";
    elements.reportPill.classList.add("ready");
  }
  if (reportPaths.technical) {
    updateReportLink("technical", reportPaths.technical);
  }
  if (reportPaths.manifest) {
    updateReportLink("manifest", reportPaths.manifest, { render: false });
  }
}

async function loadLatestSnapshotForSelectedTask() {
  const task = selectedTask();
  const requestId = ++state.snapshotRequestId;
  try {
    const response = await fetch(`/api/task-snapshot?taskSetId=${encodeURIComponent(task.id)}`);
    if (!response.ok) return;
    const snapshot = await response.json();
    if (requestId !== state.snapshotRequestId) return;
    if (state.running || state.selectedTaskId !== task.id) return;
    if (snapshot && Object.keys(snapshot).length > 0) {
      applyRunSnapshot(snapshot);
    }
  } catch (_error) {
    // Ignore bootstrap snapshot errors and keep the shell interactive.
  }
}

function sumTokens(usage) {
  if (!usage || typeof usage !== "object") return 0;
  if (typeof usage.total_tokens === "number") return usage.total_tokens;
  return Object.values(usage).reduce((sum, value) => {
    if (!value || typeof value !== "object") return sum;
    return sum + Number(value.total_tokens || 0);
  }, 0);
}

function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return "0s";
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${minutes}m ${String(rest).padStart(2, "0")}s`;
}

function updateReportLink(kind, reportPath, options = {}) {
  if (!kind || !reportPath) return;
  const shouldRenderHtml = options.render !== false && /\.md(?:own)?$/i.test(reportPath);
  const query = new URLSearchParams({ path: reportPath });
  if (shouldRenderHtml) {
    query.set("render", "html");
  }
  state.reportPaths[kind] = `/api/report?${query.toString()}`;
  syncReportLinks();
}

elements.resetButton.addEventListener("click", () => resetRun({ keepSelection: true }));
elements.clearLogButton.addEventListener("click", () => {
  elements.logFeed.innerHTML = "";
});
elements.scrollButton.addEventListener("click", () => {
  elements.logFeed.scrollTop = elements.logFeed.scrollHeight;
});
window.addEventListener("resize", () => requestAnimationFrame(renderMeasuredEdges));

renderTasks();
renderTaskDetail();
void loadLatestSnapshotForSelectedTask();
renderTree();
updateHeader();
logLine("Command deck ready.", "success");
elements.scrollButton.textContent = "Pinned bottom";
