"""Convert JSONL workflow logs into visualized artifacts.

Example:
    python visualize_json_logs.py --input artifacts/logs/demo_events.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass
class NodeRun:
    node_name: str
    event_key: str
    start_time: datetime
    end_time: datetime
    latency_ms: int
    status: str
    dataset: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize workflow JSONL logs.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL log file (for example artifacts/logs/demo_events.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directory where output figures and summaries are written. "
            "If omitted and --input is under artifacts/runs/<run_label>/logs/, "
            "outputs are written to artifacts/runs/<run_label>/figures/."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=["latest", "all"],
        default="latest",
        help="Use only latest workflow run or all runs in the file.",
    )
    return parser.parse_args()


def parse_iso_timestamp(raw: str) -> datetime | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def extract_node_runs(events: list[dict[str, Any]]) -> list[NodeRun]:
    starts: dict[str, tuple[str, datetime, str]] = {}
    runs: list[NodeRun] = []

    for event in events:
        event_type = str(event.get("event") or "")
        node_name = str(event.get("node_name") or "")
        event_key = str(event.get("event_key") or node_name)
        dataset = str(event.get("dataset") or "")
        ts = parse_iso_timestamp(str(event.get("timestamp") or ""))
        if ts is None:
            continue

        if event_type == "node_start" and node_name:
            starts[event_key] = (node_name, ts, dataset)
            continue

        if event_type != "node_result" or not node_name:
            continue
        start_info = starts.get(event_key)
        if not start_info:
            continue
        start_node_name, start_time, start_dataset = start_info
        latency_raw = event.get("latency_ms")
        if isinstance(latency_raw, (int, float)):
            latency_ms = int(latency_raw)
        else:
            latency_ms = int((ts - start_time).total_seconds() * 1000)
        status = str(event.get("status") or "unknown")
        runs.append(
            NodeRun(
                node_name=start_node_name,
                event_key=event_key,
                start_time=start_time,
                end_time=ts,
                latency_ms=latency_ms,
                status=status,
                dataset=start_dataset or dataset,
            )
        )
    return runs


def filter_runs_by_scope(events: list[dict[str, Any]], runs: list[NodeRun], scope: str) -> list[NodeRun]:
    if scope == "all":
        return runs

    latest_workflow_start: datetime | None = None
    latest_workflow_end: datetime | None = None
    for event in events:
        event_type = str(event.get("event") or "")
        ts = parse_iso_timestamp(str(event.get("timestamp") or ""))
        if ts is None:
            continue
        if event_type == "workflow_start":
            latest_workflow_start = ts
            latest_workflow_end = None
        elif event_type == "workflow_end" and latest_workflow_start is not None and ts >= latest_workflow_start:
            latest_workflow_end = ts

    if latest_workflow_start is None:
        return runs

    filtered: list[NodeRun] = []
    for run in runs:
        if run.start_time < latest_workflow_start:
            continue
        if latest_workflow_end is not None and run.end_time > latest_workflow_end:
            continue
        filtered.append(run)
    return filtered or runs


def extract_token_usage_by_module(events: list[dict[str, Any]], scope: str) -> dict[str, int]:
    token_usage: dict[str, int] = {}
    workflow_end_events = [event for event in events if str(event.get("event")) == "workflow_end"]
    target_end: dict[str, Any] | None = workflow_end_events[-1] if workflow_end_events else None

    if scope == "all":
        target_end = None

    if target_end is not None:
        usage = target_end.get("token_usage_by_module")
        if isinstance(usage, dict):
            for module, payload in usage.items():
                if isinstance(payload, dict):
                    total = payload.get("total_tokens")
                    if isinstance(total, (int, float)):
                        token_usage[str(module)] = int(total)
        if token_usage:
            return token_usage

    # Fallback: use latest token_usage snapshot found on node_result entries.
    latest_by_node: dict[str, int] = {}
    for event in events:
        if str(event.get("event")) != "node_result":
            continue
        node_name = str(event.get("node_name") or "")
        usage = event.get("token_usage")
        if not node_name or not isinstance(usage, dict):
            continue
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)):
            latest_by_node[node_name] = int(total)
    return latest_by_node


def extract_module_usage_details(events: list[dict[str, Any]], scope: str) -> dict[str, dict[str, int]]:
    """Extract per-module input/output token totals and recent prompt lengths."""
    workflow_end_events = [event for event in events if str(event.get("event")) == "workflow_end"]
    if not workflow_end_events:
        return {}
    target_events = workflow_end_events if scope == "all" else [workflow_end_events[-1]]

    merged: dict[str, dict[str, int]] = {}
    for event in target_events:
        usage = event.get("token_usage_by_module")
        if not isinstance(usage, dict):
            continue
        for module, payload in usage.items():
            if not isinstance(payload, dict):
                continue
            bucket = merged.setdefault(
                str(module),
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "avg_prompt_chars": 0},
            )
            input_tokens = int(payload.get("input_tokens") or 0)
            output_tokens = int(payload.get("output_tokens") or 0)
            total_tokens = int(payload.get("total_tokens") or (input_tokens + output_tokens))
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
            bucket["total_tokens"] += total_tokens

            recent = payload.get("recent_interactions")
            if isinstance(recent, list) and recent:
                prompt_lengths: list[int] = []
                for item in recent:
                    if not isinstance(item, dict):
                        continue
                    messages = item.get("input")
                    if not isinstance(messages, list):
                        continue
                    total_chars = 0
                    for msg in messages:
                        if isinstance(msg, dict):
                            total_chars += len(str(msg.get("content") or ""))
                    if total_chars:
                        prompt_lengths.append(total_chars)
                if prompt_lengths:
                    bucket["avg_prompt_chars"] = int(sum(prompt_lengths) / len(prompt_lengths))
    return merged


def write_summary_csv(runs: list[NodeRun], output_path: Path) -> None:
    lines = ["node_name,count,avg_latency_ms,max_latency_ms,total_latency_ms,error_count"]
    grouped: dict[str, list[NodeRun]] = {}
    for run in runs:
        grouped.setdefault(run.node_name, []).append(run)

    for node_name in sorted(grouped):
        items = grouped[node_name]
        latencies = [item.latency_ms for item in items]
        error_count = sum(1 for item in items if item.status == "error")
        lines.append(
            ",".join(
                [
                    node_name,
                    str(len(items)),
                    str(int(sum(latencies) / len(latencies))),
                    str(max(latencies)),
                    str(sum(latencies)),
                    str(error_count),
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_node_latency_bar(runs: list[NodeRun], output_path: Path) -> None:
    grouped: dict[str, list[int]] = {}
    for run in runs:
        grouped.setdefault(run.node_name, []).append(run.latency_ms)

    nodes = sorted(grouped)
    avg_latency = [sum(grouped[node]) / len(grouped[node]) for node in nodes]

    fig, ax = plt.subplots(figsize=(12, max(4, len(nodes) * 0.5)))
    bars = ax.barh(nodes, avg_latency, color="#4C78A8")
    ax.set_title("Average Node Latency")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Node")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, avg_latency):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {int(value)}ms", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_waterfall(runs: list[NodeRun], output_path: Path) -> None:
    sorted_runs = sorted(runs, key=lambda item: item.start_time)
    labels = [f"{run.node_name}#{idx + 1}" for idx, run in enumerate(sorted_runs)]
    first_start = sorted_runs[0].start_time
    starts = [(run.start_time - first_start).total_seconds() for run in sorted_runs]
    raw_widths = [(run.end_time - run.start_time).total_seconds() for run in sorted_runs]
    widths = [max(width, 0.001) for width in raw_widths]
    colors = ["#E45756" if run.status == "error" else "#72B7B2" for run in sorted_runs]

    fig_height = max(5, len(sorted_runs) * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    y_positions = list(range(len(sorted_runs)))
    ax.barh(y_positions, widths, left=starts, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Workflow Waterfall Timeline")
    ax.set_xlabel("Seconds Since Workflow Start")
    ax.set_ylabel("Node Run")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    max_end = max(start + width for start, width in zip(starts, widths))
    ax.set_xlim(0, max_end * 1.05)

    for idx, (start, width, raw_width) in enumerate(zip(starts, widths, raw_widths)):
        if raw_width >= 1:
            label = f"{raw_width:.2f}s"
        else:
            label = f"{int(raw_width * 1000)}ms"
        text_x = start + width + max(0.2, max_end * 0.003)
        text_x = min(text_x, max_end * 1.02)
        ax.text(text_x, idx, label, va="center", ha="left", fontsize=8, color="#333333")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_token_usage_by_module(token_usage: dict[str, int], output_path: Path) -> None:
    if not token_usage:
        return
    modules = sorted(token_usage)
    totals = [token_usage[module] for module in modules]

    fig, ax = plt.subplots(figsize=(12, max(4, len(modules) * 0.5)))
    bars = ax.barh(modules, totals, color="#F28E2B")
    ax.set_title("Token Usage by Module")
    ax.set_xlabel("Total Tokens")
    ax.set_ylabel("Module")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, total in zip(bars, totals):
        ax.text(total, bar.get_y() + bar.get_height() / 2, f" {total}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_token_io_and_prompt(
    module_usage: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    if not module_usage:
        return
    modules = sorted(module_usage)
    input_tokens = [module_usage[module]["input_tokens"] for module in modules]
    output_tokens = [module_usage[module]["output_tokens"] for module in modules]
    prompt_chars = [module_usage[module]["avg_prompt_chars"] for module in modules]

    fig, ax = plt.subplots(figsize=(12, max(4, len(modules) * 0.55)))
    y_pos = list(range(len(modules)))
    ax.barh(y_pos, input_tokens, color="#4C78A8", label="Input tokens")
    ax.barh(y_pos, output_tokens, left=input_tokens, color="#F28E2B", label="Output tokens")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modules)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Module")
    ax.set_title("Input/Output Tokens and Prompt Size by Module")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    ax2 = ax.twiny()
    ax2.plot(prompt_chars, y_pos, "o-", color="#59A14F", linewidth=1.4, label="Avg prompt chars")
    ax2.set_xlabel("Average prompt length (chars)")
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input log file not found: {input_path}")

    events = load_events(input_path)
    runs = extract_node_runs(events)
    runs = filter_runs_by_scope(events, runs, args.scope)
    if not runs:
        raise RuntimeError("No node runs found in JSONL log. Check input format.")

    if args.output_dir.strip():
        output_dir = Path(args.output_dir).expanduser()
    elif input_path.parent.name == "logs" and input_path.parent.parent.name:
        output_dir = input_path.parent.parent / "figures"
    else:
        output_dir = Path("artifacts/figures").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    summary_csv = output_dir / f"{stem}_summary.csv"
    bar_png = output_dir / f"{stem}_node_latency.png"
    waterfall_png = output_dir / f"{stem}_waterfall.png"
    token_png = output_dir / f"{stem}_token_usage.png"
    token_io_png = output_dir / f"{stem}_token_io_prompt.png"

    write_summary_csv(runs, summary_csv)
    plot_node_latency_bar(runs, bar_png)
    plot_waterfall(runs, waterfall_png)
    token_usage = extract_token_usage_by_module(events, args.scope)
    plot_token_usage_by_module(token_usage, token_png)
    module_usage = extract_module_usage_details(events, args.scope)
    plot_token_io_and_prompt(module_usage, token_io_png)

    print(f"Summary CSV: {summary_csv}")
    print(f"Node latency chart: {bar_png}")
    print(f"Waterfall chart: {waterfall_png}")
    if token_usage:
        print(f"Token usage chart: {token_png}")
        if module_usage:
            print(f"Token I/O + prompt chart: {token_io_png}")
    else:
        print("Token usage chart: skipped (no token usage found in input log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

