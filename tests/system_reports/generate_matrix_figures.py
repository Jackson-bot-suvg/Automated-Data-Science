from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "spaceship_titanic": "Spaceship Titanic",
    "academic_success": "Academic Success",
    "house_prices": "House Prices",
    "plate_defect": "Plate Defect",
    "titanic_autokaggle": "Titanic AutoKaggle",
    "bank_churn": "Bank Churn",
    "ghouls": "Ghouls",
    "obesity_risks": "Obesity Risks",
}

PHASE_ORDER = [
    "preliminary_eda",
    "data_cleaning",
    "indepth_eda",
    "feature_engineering",
    "model_building",
]

MEASURED_NODE_ORDER = [
    "reader",
    "planner",
    "preliminary_eda",
    "data_cleaning",
    "indepth_eda",
    "feature_engineering",
    "model_building",
]


@dataclass
class TaskResult:
    name: str
    group: str
    dataset_type: str
    task_family: str
    target_mode: str
    total_runtime_s: float
    metric_name: str
    metric_value: str
    phase_gates: list[dict[str, str]]
    node_timings: list[dict[str, str]]

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.name, self.name.replace("_", " ").title())

    @property
    def metric_numeric(self) -> float | None:
        try:
            return float(self.metric_value)
        except ValueError:
            return None


def parse_markdown_table(section: str, header_name: str) -> list[dict[str, str]]:
    marker = f"### {header_name}"
    start = section.find(marker)
    if start == -1:
        return []
    remaining = section[start + len(marker) :]
    lines = remaining.splitlines()
    table_lines: list[str] = []
    started = False
    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
            started = True
        elif started:
            break
    if len(table_lines) < 3:
        return []
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for raw_line in table_lines[2:]:
        cells = [cell.strip() for cell in raw_line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def parse_report(report_path: Path) -> list[TaskResult]:
    text = report_path.read_text(encoding="utf-8")

    group_map: dict[str, str] = {}
    for match in re.finditer(r"^- `([^`]+)`: .* \[(primary|smoke)\] ", text, flags=re.MULTILINE):
        group_map[match.group(1)] = match.group(2)

    matches = list(re.finditer(r"^## \d+\. `([^`]+)`", text, flags=re.MULTILINE))
    tasks: list[TaskResult] = []

    for idx, match in enumerate(matches):
        name = match.group(1)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else text.find("\n## 结束信息")
        if end == -1:
            end = len(text)
        section = text[start:end]

        dataset_type = re.search(r"- 数据集类型：`([^`]+)`", section)
        task_info = re.search(r"- 任务族 / 目标模式：`([^`]+)` / `([^`]+)`", section)
        total_runtime = re.search(r"- 总耗时：`([\d.]+)` s", section)
        metric = re.search(r"- 主指标：`([^`]+)` = `([^`]+)`", section)

        if not (dataset_type and task_info and total_runtime and metric):
            continue

        tasks.append(
            TaskResult(
                name=name,
                group=group_map.get(name, "unknown"),
                dataset_type=dataset_type.group(1),
                task_family=task_info.group(1),
                target_mode=task_info.group(2),
                total_runtime_s=float(total_runtime.group(1)),
                metric_name=metric.group(1),
                metric_value=metric.group(2),
                phase_gates=parse_markdown_table(section, "Phase Gates"),
                node_timings=parse_markdown_table(section, "Node Timings"),
            )
        )

    return tasks


def node_time(task: TaskResult, node_name: str) -> float:
    for row in task.node_timings:
        if row.get("Node", "").strip("`") == node_name:
            raw = row.get("Time (s)", "-")
            return 0.0 if raw == "-" else float(raw)
    return 0.0


def phase_attempt(task: TaskResult, phase_name: str) -> int:
    for row in task.phase_gates:
        if row.get("Phase", "").strip("`") == phase_name:
            return int(row.get("Attempt", "0"))
    return 0


def phase_retry(task: TaskResult, phase_name: str) -> int:
    for row in task.phase_gates:
        if row.get("Phase", "").strip("`") == phase_name:
            return int(row.get("Retries", "0"))
    return 0


def sort_tasks(tasks: list[TaskResult]) -> list[TaskResult]:
    return tasks


def group_color(group: str) -> str:
    return {"primary": "#1f77b4", "smoke": "#ff7f0e"}.get(group, "#7f7f7f")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_1_total_runtime(tasks: list[TaskResult], out_dir: Path) -> Path:
    ordered = sorted(tasks, key=lambda t: t.total_runtime_s, reverse=True)
    labels = [task.display_name for task in ordered]
    values = [task.total_runtime_s for task in ordered]
    colors = [group_color(task.group) for task in ordered]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor="#333333")
    ax.invert_yaxis()
    ax.set_xlabel("Total runtime (seconds)")
    ax.set_title("Figure 1. End-to-End Workflow Runtime Across Evaluation Datasets")
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + 3, bar.get_y() + bar.get_height() / 2, f"{value:.1f}s", va="center", fontsize=9)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=group_color("primary"), lw=8, label="Primary dataset"),
            plt.Line2D([0], [0], color=group_color("smoke"), lw=8, label="Smoke dataset"),
        ],
        loc="lower right",
    )
    path = out_dir / "figure_01_total_runtime.png"
    save_figure(fig, path)
    return path


def figure_2_runtime_decomposition(tasks: list[TaskResult], out_dir: Path) -> Path:
    labels = [task.display_name for task in tasks]
    reader = np.array([node_time(task, "reader") for task in tasks])
    planner = np.array([node_time(task, "planner") for task in tasks])
    modeling = np.array([node_time(task, "model_building") for task in tasks])
    other = np.array(
        [
            max(task.total_runtime_s - r - p - m, 0.0)
            for task, r, p, m in zip(tasks, reader, planner, modeling, strict=True)
        ]
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(labels, reader, label="Reader", color="#4c78a8")
    ax.bar(labels, planner, bottom=reader, label="Planner", color="#f58518")
    ax.bar(labels, modeling, bottom=reader + planner, label="Model Building", color="#54a24b")
    ax.bar(labels, other, bottom=reader + planner + modeling, label="Other Measured/Unmeasured Time", color="#bdbdbd")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Figure 2. Runtime Decomposition by Major Workflow Component")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")
    path = out_dir / "figure_02_runtime_decomposition.png"
    save_figure(fig, path)
    return path


def figure_3_planner_vs_modeling(tasks: list[TaskResult], out_dir: Path) -> Path:
    labels = [task.display_name for task in tasks]
    planner = np.array([node_time(task, "planner") for task in tasks])
    modeling = np.array([node_time(task, "model_building") for task in tasks])
    x = np.arange(len(tasks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, planner, width=width, label="Planner", color="#f58518")
    ax.bar(x + width / 2, modeling, width=width, label="Model Building", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Figure 3. Planner and Model-Building Runtime by Dataset")
    ax.legend(loc="upper right")
    path = out_dir / "figure_03_planner_vs_modeling.png"
    save_figure(fig, path)
    return path


def figure_4_runtime_share(tasks: list[TaskResult], out_dir: Path) -> Path:
    labels = [task.display_name for task in tasks]
    planner_share = np.array([node_time(task, "planner") / task.total_runtime_s for task in tasks])
    modeling_share = np.array([node_time(task, "model_building") / task.total_runtime_s for task in tasks])
    other_share = np.clip(1.0 - planner_share - modeling_share, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, planner_share * 100, label="Planner", color="#f58518")
    ax.bar(labels, modeling_share * 100, bottom=planner_share * 100, label="Model Building", color="#54a24b")
    ax.bar(
        labels,
        other_share * 100,
        bottom=(planner_share + modeling_share) * 100,
        label="Other",
        color="#bdbdbd",
    )
    ax.set_ylabel("Share of total runtime (%)")
    ax.set_title("Figure 4. Planner and Model-Building Shares of End-to-End Runtime")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")
    path = out_dir / "figure_04_runtime_share.png"
    save_figure(fig, path)
    return path


def draw_heatmap(
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    colorbar_label: str,
    out_path: Path,
    cmap: str = "Blues",
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="#111111", fontsize=9)
    save_figure(fig, out_path)
    return out_path


def figure_5_phase_attempts(tasks: list[TaskResult], out_dir: Path) -> Path:
    matrix = np.array([[phase_attempt(task, phase) for phase in PHASE_ORDER] for task in tasks], dtype=float)
    return draw_heatmap(
        matrix,
        PHASE_ORDER,
        [task.display_name for task in tasks],
        "Figure 5. Phase Attempt Counts Across Evaluation Datasets",
        "Attempts",
        out_dir / "figure_05_phase_attempts.png",
        cmap="PuBu",
    )


def figure_6_phase_retries(tasks: list[TaskResult], out_dir: Path) -> Path:
    matrix = np.array([[phase_retry(task, phase) for phase in PHASE_ORDER] for task in tasks], dtype=float)
    return draw_heatmap(
        matrix,
        PHASE_ORDER,
        [task.display_name for task in tasks],
        "Figure 6. Phase Retry Profile Across Evaluation Datasets",
        "Retries",
        out_dir / "figure_06_phase_retries.png",
        cmap="OrRd",
    )


def figure_7_metric_summary(tasks: list[TaskResult], out_dir: Path) -> Path:
    classification_tasks = [
        task for task in tasks if task.metric_numeric is not None and task.metric_name in {"accuracy", "f1_macro", "roc_auc"}
    ]
    regression_tasks = [
        task for task in tasks if task.metric_numeric is not None and task.metric_name == "neg_root_mean_squared_error"
    ]
    unavailable_tasks = [task for task in tasks if task.metric_numeric is None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2.2, 1.0, 0.8]})

    cls_labels = [task.display_name for task in classification_tasks]
    cls_values = [task.metric_numeric for task in classification_tasks]
    cls_colors = ["#4c78a8" if task.metric_name == "accuracy" else "#54a24b" if task.metric_name == "f1_macro" else "#f58518" for task in classification_tasks]
    axes[0].barh(cls_labels, cls_values, color=cls_colors, edgecolor="#333333")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.0)
    axes[0].set_xlabel("Score")
    axes[0].set_title("Classification Tasks")

    rmse_labels = [task.display_name for task in regression_tasks]
    rmse_values = [abs(task.metric_numeric) for task in regression_tasks]
    axes[1].barh(rmse_labels, rmse_values, color="#b279a2", edgecolor="#333333")
    if rmse_labels:
        axes[1].invert_yaxis()
    axes[1].set_xlabel("RMSE")
    axes[1].set_title("Regression Tasks")

    axes[2].axis("off")
    lines = ["Metric availability"]
    for task in unavailable_tasks:
        lines.append(f"{task.display_name}: {task.metric_name} not available")
    if not unavailable_tasks:
        lines.append("All primary metrics available")
    axes[2].text(0.0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10)

    fig.suptitle("Figure 7. Primary Validation Metrics by Task Family", fontsize=14, y=1.02)
    path = out_dir / "figure_07_metric_summary.png"
    save_figure(fig, path)
    return path


def figure_8_mean_node_runtime(tasks: list[TaskResult], out_dir: Path) -> Path:
    means = []
    stds = []
    labels = []
    for node in MEASURED_NODE_ORDER:
        values = [node_time(task, node) for task in tasks]
        labels.append(node.replace("_", " ").title())
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.barh(labels, means, xerr=stds, color="#4c78a8", edgecolor="#333333", ecolor="#333333", capsize=4)
    ax.set_xlabel("Mean runtime across datasets (seconds)")
    ax.set_title("Figure 8. Mean Measured Node Runtime Across the Benchmark Matrix")
    ax.invert_yaxis()
    path = out_dir / "figure_08_mean_node_runtime.png"
    save_figure(fig, path)
    return path


def generate_figures(report_path: Path) -> list[Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    tasks = sort_tasks(parse_report(report_path))
    out_dir = report_path.parent / f"{report_path.stem}_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        figure_1_total_runtime(tasks, out_dir),
        figure_2_runtime_decomposition(tasks, out_dir),
        figure_3_planner_vs_modeling(tasks, out_dir),
        figure_4_runtime_share(tasks, out_dir),
        figure_5_phase_attempts(tasks, out_dir),
        figure_6_phase_retries(tasks, out_dir),
        figure_7_metric_summary(tasks, out_dir),
        figure_8_mean_node_runtime(tasks, out_dir),
    ]
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate academic-ready figures from a system matrix report.")
    parser.add_argument("report", type=Path, help="Path to the system matrix markdown report.")
    args = parser.parse_args()
    figures = generate_figures(args.report)
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
