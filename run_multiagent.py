"""Run the MultiAgent LangGraph workflow with live node progress logs.

Usage:
    python run_multiagent.py spaceship_titanic
    python run_multiagent.py plate_defect
    python run_multiagent.py --task "predict customer churn" --dataset-path data/bank_churn

The script runs the full workflow by default, including business translation.
Use --stop-after-technical-summary to end after the Technical Summarizer node.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.intake import build_initial_state
from app.schemas.state_schema import GraphState
from app.validation_matrix import ALL_VALIDATION_DATASETS, PRIMARY_ACCEPTANCE_DATASETS


NODE_DESCRIPTIONS: dict[str, str] = {
    "reader": "Reading task configuration, dataset paths, and basic dataset metadata.",
    "background_reviewer": "Checking whether the reader output is complete enough to start planning.",
    "planner": "Choosing the next workflow phase from the shared GraphState.",
    "preliminary_eda": "Profiling the raw dataset and writing initial EDA artifacts.",
    "data_cleaning": "Planning and applying data cleaning tool calls.",
    "indepth_eda": "Running deeper exploratory analysis to extract more insights from the dataset.",
    "feature_engineering": "Planning and applying feature transformation tool calls.",
    "model_building": "Training candidate models, selecting the best model, and saving predictions.",
    "reviewer": "Reviewing the latest phase outputs and deciding approve, retry, or abort.",
    "summarizer": "Generating the final Technical Report after model building is approved.",
    "business_translator": "Translating the technical report into a business-facing report.",
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    task: str
    dataset: str
    target: str = ""
    metric_hint: str = ""
    description: str = ""


PRESET_RUNS: dict[str, RunSpec] = {
    spec.key: RunSpec(
        key=spec.key,
        task=spec.task,
        dataset=spec.dataset,
        target=spec.target,
        metric_hint=spec.metric_hint,
        description=spec.description,
    )
    for spec in ALL_VALIDATION_DATASETS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a supported MultiAgent LangGraph workflow and print live node logs."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Preset dataset key to run, or a dataset path when paired with --task.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Natural-language task description for a custom dataset-path run.",
    )
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Explicit dataset directory or CSV path for a custom run.",
    )
    parser.add_argument(
        "--target",
        default="",
        help="Optional target column or target hint for the run.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print the supported preset dataset keys and exit.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the environment file containing LLM configuration.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=80,
        help="LangGraph recursion limit for the workflow run.",
    )
    parser.add_argument(
        "--stop-after-technical-summary",
        action="store_true",
        help="Stop after the technical summarizer instead of running business translation.",
    )
    return parser.parse_args()


def resolve_run_spec(args: argparse.Namespace) -> RunSpec:
    preset_key = (args.dataset or "").strip().lower()
    custom_task = (args.task or "").strip()
    custom_dataset = (args.dataset_path or "").strip()
    custom_target = (args.target or "").strip()

    if preset_key and preset_key in PRESET_RUNS and not custom_task and not custom_dataset:
        return PRESET_RUNS[preset_key]

    dataset_path = custom_dataset or (args.dataset or "").strip()
    if custom_task and dataset_path:
        return RunSpec(
            key="custom",
            task=custom_task,
            dataset=dataset_path,
            target=custom_target,
            metric_hint="",
            description="custom validation run",
        )

    supported = ", ".join(sorted(PRESET_RUNS))
    raise ValueError(
        "Please choose a preset dataset key or provide both --task and "
        f"--dataset-path. Supported presets: {supported}."
    )


def print_supported_datasets() -> None:
    primary_keys = {spec.key for spec in PRIMARY_ACCEPTANCE_DATASETS}
    print("Supported preset datasets:")
    for key in sorted(PRESET_RUNS):
        spec = PRESET_RUNS[key]
        tier = "primary" if key in primary_keys else "smoke"
        print(f"- {key}: {spec.description} [{tier}] [{spec.dataset}]")


def configure_runtime(env_file: str) -> None:
    load_dotenv(env_file)
    mpl_cache = Path(os.environ.get("MPLCONFIGDIR", "/tmp/multiagent-mpl-cache"))
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    warnings.filterwarnings("ignore", category=FutureWarning)


def log_node_start(node_name: str) -> None:
    description = NODE_DESCRIPTIONS.get(node_name, "Running this LangGraph node.")
    print(f"[RUNNING] LangGraph node: {node_name} - {description}", flush=True)


def log_node_result(node_name: str, result: Any) -> dict[str, str]:
    if not isinstance(result, dict):
        print(f"[DONE] {node_name} completed.", flush=True)
        return {"technical_report_path": "", "business_report_path": ""}

    parts: list[str] = []
    current_phase = result.get("current_phase")
    if current_phase:
        parts.append(f"phase={current_phase}")

    target_spec = result.get("target_spec")
    if isinstance(target_spec, dict):
        target_columns = target_spec.get("target_columns") or []
        if target_columns:
            parts.append(f"targets={','.join(str(col) for col in target_columns)}")
        if target_spec.get("target_mode"):
            parts.append(f"target_mode={target_spec['target_mode']}")
        if target_spec.get("task_family"):
            parts.append(f"task_family={target_spec['task_family']}")

    review_state = result.get("review_state")
    if isinstance(review_state, dict):
        status = review_state.get("latest_review_status")
        pending_phase = review_state.get("pending_review_phase")
        if status:
            detail = f"review={status}"
            if pending_phase:
                detail += f"({pending_phase})"
            parts.append(detail)

    report_state = result.get("report_state")
    technical_report_path = ""
    business_report_path = ""
    if isinstance(report_state, dict):
        technical_report_path = str(report_state.get("technical_report_path") or "")
        if technical_report_path:
            parts.append(f"technical_report={technical_report_path}")
        business_report_path = str(report_state.get("business_report_path") or "")
        if business_report_path:
            parts.append(f"business_report={business_report_path}")

    runtime_state = result.get("runtime_state")
    if isinstance(runtime_state, dict):
        phase_runtime = runtime_state.get("phase_runtime_sec") or {}
        if isinstance(phase_runtime, dict) and node_name in phase_runtime:
            parts.append(f"duration={float(phase_runtime[node_name]):.4f}s")

    suffix = f" [{', '.join(parts)}]" if parts else ""
    print(f"[DONE] {node_name} completed.{suffix}", flush=True)
    return {
        "technical_report_path": technical_report_path,
        "business_report_path": business_report_path,
    }


def build_run_state(spec: RunSpec) -> GraphState:
    return build_initial_state(
        task=spec.task,
        dataset=spec.dataset,
        target=spec.target,
        metric_hint=spec.metric_hint,
        workspace_root=".",
        thread_id=f"{spec.key}-cli",
    )


async def run_dataset(
    spec: RunSpec,
    recursion_limit: int,
    *,
    stop_after_technical_summary: bool,
) -> tuple[Path, Path | None]:
    from app.graph.workflow import build_workflow

    graph = build_workflow(
        stop_after_technical_summary=stop_after_technical_summary
    )
    state = build_run_state(spec)
    expected_report_path = str(state.artifacts.get("technical_report") or "")
    expected_business_report_path = str(state.artifacts.get("business_report") or "")
    timing_report_path = _timing_report_path(state)
    technical_report_path = ""
    business_report_path = ""
    latest_phase_durations: dict[str, float] = {}

    print(f"Starting MultiAgent LangGraph workflow for dataset: {spec.key}", flush=True)
    print(f"Task: {spec.task}", flush=True)
    print(f"Dataset path: {spec.dataset}", flush=True)
    if spec.target:
        print(f"Target hint: {spec.target}", flush=True)
    if stop_after_technical_summary:
        print(
            "The workflow will stop after the Technical Summarizer node.",
            flush=True,
        )
    else:
        print(
            "The workflow will continue through the Business Translator node.",
            flush=True,
        )

    async for event in graph.astream(state, stream_mode="debug", config={"recursion_limit": recursion_limit}):
        event_type = event.get("type")
        payload = event.get("payload") or {}
        node_name = str(payload.get("name") or "")

        if event_type == "task" and node_name:
            log_node_start(node_name)
        elif event_type == "task_result" and node_name:
            if payload.get("error"):
                print(f"[ERROR] {node_name} failed: {payload['error']}", flush=True)
                continue
            result = payload.get("result")
            report_paths = log_node_result(node_name, result)
            latest_phase_durations = _extract_phase_durations(result, latest_phase_durations)
            technical_report_path = (
                report_paths.get("technical_report_path") or technical_report_path
            )
            business_report_path = (
                report_paths.get("business_report_path") or business_report_path
            )

    if not technical_report_path:
        technical_report_path = expected_report_path
    if not business_report_path:
        business_report_path = expected_business_report_path

    report_path = Path(technical_report_path).expanduser()
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    report_path = report_path.resolve()
    if not report_path.exists():
        raise FileNotFoundError(
            f"Technical Report was not produced for this run: {report_path}"
        )
    business_report = None
    if business_report_path:
        candidate = Path(business_report_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        if candidate.exists():
            business_report = candidate
    _write_timing_report(
        timing_report_path=timing_report_path,
        spec=spec,
        run_state=state,
        phase_durations=latest_phase_durations,
        technical_report_path=report_path,
        business_report_path=business_report,
    )
    print(f"Timing report: {timing_report_path}", flush=True)
    return report_path, business_report


def _timing_report_path(state: GraphState) -> Path:
    artifact_root = Path(state.runtime_state.artifact_root or "artifacts")
    if not artifact_root.is_absolute():
        artifact_root = Path.cwd() / artifact_root
    target = artifact_root / "reports" / "run_timing.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def _extract_phase_durations(result: Any, existing: dict[str, float]) -> dict[str, float]:
    updated = dict(existing)
    if not isinstance(result, dict):
        return updated
    runtime_state = result.get("runtime_state")
    if not isinstance(runtime_state, dict):
        return updated
    phase_runtime = runtime_state.get("phase_runtime_sec")
    if not isinstance(phase_runtime, dict):
        return updated
    for key, value in phase_runtime.items():
        try:
            updated[str(key)] = round(float(value), 4)
        except Exception:
            continue
    return updated


def _write_timing_report(
    *,
    timing_report_path: Path,
    spec: RunSpec,
    run_state: GraphState,
    phase_durations: dict[str, float],
    technical_report_path: Path,
    business_report_path: Path | None,
) -> None:
    payload = {
        "dataset_key": spec.key,
        "task": spec.task,
        "dataset_path": spec.dataset,
        "target_hint": spec.target,
        "thread_id": run_state.runtime_state.thread_id,
        "run_id": run_state.runtime_state.run_id,
        "artifact_root": run_state.runtime_state.artifact_root,
        "technical_report_path": str(technical_report_path),
        "business_report_path": str(business_report_path) if business_report_path else "",
        "phase_durations_sec": phase_durations,
        "total_runtime_sec": round(sum(phase_durations.values()), 4),
    }
    timing_report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def print_technical_report(report_path: Path) -> None:
    if not report_path.exists():
        raise FileNotFoundError(f"Technical Report was not found at {report_path}")

    print("\n" + "=" * 80)
    print("TECHNICAL REPORT")
    print("=" * 80)
    print(report_path.read_text(encoding="utf-8").rstrip() + "\n")


async def async_main() -> int:
    args = parse_args()
    try:
        if args.list_datasets:
            print_supported_datasets()
            return 0

        spec = resolve_run_spec(args)
        configure_runtime(args.env_file)
        technical_report_path, business_report_path = await run_dataset(
            spec,
            args.recursion_limit,
            stop_after_technical_summary=args.stop_after_technical_summary,
        )
        print_technical_report(technical_report_path)
        if business_report_path:
            print(f"Business report: {business_report_path}")
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nWorkflow failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
