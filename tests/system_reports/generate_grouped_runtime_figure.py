from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = [
    ("Binary Classification", ["spaceship_titanic", "titanic_autokaggle", "bank_churn"]),
    ("Multiclass Classification", ["academic_success", "ghouls", "obesity_risks"]),
    ("Regression", ["house_prices"]),
    ("Multi-Target", ["plate_defect"]),
]


def load_parser_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("matrix_figures", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load parser helpers from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def node_count(task, node_name: str) -> int:
    for row in task.node_timings:
        if row.get("Node", "").strip("`") == node_name:
            return int(row.get("Count", "0"))
    return 0


def build_runtime_segments(task, node_time_fn):
    reader = node_time_fn(task, "reader")
    planner = node_time_fn(task, "planner")
    data_feature = sum(
        [
            node_time_fn(task, "preliminary_eda"),
            node_time_fn(task, "data_cleaning"),
            node_time_fn(task, "indepth_eda"),
            node_time_fn(task, "feature_engineering"),
        ]
    )
    model = node_time_fn(task, "model_building")
    summarizer = node_time_fn(task, "summarizer")

    measured = reader + planner + data_feature + model + summarizer
    residual = max(task.total_runtime_s - measured, 0.0)

    review_calls = node_count(task, "reviewer")
    bg_calls = node_count(task, "background_reviewer")
    review_gatekeeping = min((0.75 * review_calls) + (0.5 * bg_calls), residual * 0.20) if residual > 0 else 0.0
    report_generation = max(residual - review_gatekeeping, 0.0)

    return {
        "Reader": reader,
        "Planner": planner,
        "Data and Feature Processing": data_feature,
        "Model Building": model,
        "Review and Gatekeeping": review_gatekeeping,
        "Report Generation": report_generation,
    }


def generate_figure(report_path: Path) -> Path:
    helper_script = report_path.parent / "generate_matrix_figures.py"
    helpers = load_parser_module(helper_script)
    tasks = {task.name: task for task in helpers.parse_report(report_path)}

    ordered_tasks = []
    for _, names in GROUP_ORDER:
        for name in names:
            if name in tasks:
                ordered_tasks.append(tasks[name])

    labels = [task.display_name for task in ordered_tasks]
    segments = [build_runtime_segments(task, helpers.node_time) for task in ordered_tasks]

    segment_names = [
        "Reader",
        "Planner",
        "Data and Feature Processing",
        "Model Building",
        "Review and Gatekeeping",
        "Report Generation",
    ]
    colors = {
        "Reader": "#4c78a8",
        "Planner": "#f58518",
        "Data and Feature Processing": "#72b7b2",
        "Model Building": "#54a24b",
        "Review and Gatekeeping": "#b279a2",
        "Report Generation": "#bab0ac",
    }

    x = np.arange(len(ordered_tasks))
    width = 0.72

    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    bottom = np.zeros(len(ordered_tasks))

    for segment_name in segment_names:
        values = np.array([seg[segment_name] for seg in segments])
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            label=segment_name,
            color=colors[segment_name],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += values

    for idx, task in enumerate(ordered_tasks):
        ax.text(
            x[idx],
            task.total_runtime_s + 4,
            f"{task.total_runtime_s:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("End-to-end runtime (seconds)")
    ax.set_xlabel("Dataset")
    ax.set_title("End-to-End Runtime Composition by Dataset and Task Family", fontsize=15, pad=18)

    start = 0
    ymax = max(task.total_runtime_s for task in ordered_tasks) * 1.16
    for group_name, names in GROUP_ORDER:
        present = [name for name in names if name in tasks]
        if not present:
            continue
        end = start + len(present) - 1
        center = (start + end) / 2
        ax.text(center, ymax * 0.985, group_name, ha="center", va="top", fontsize=9, fontweight="bold")
        if end < len(ordered_tasks) - 1:
            ax.axvline(end + 0.5, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
        start = end + 1

    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    out_path = report_path.parent / "system-matrix-20260430-125627-runtime-by-task-family.png"
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped runtime figure for the system matrix report.")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    out = generate_figure(args.report)
    print(out)


if __name__ == "__main__":
    main()
