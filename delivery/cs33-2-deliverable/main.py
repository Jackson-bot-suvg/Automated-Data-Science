"""Natural-language CLI entry point for the MultiAgent workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.agents.bg_reviewer import run_bg_reviewer
from app.agents.reader import run_reader
from app.intake import build_initial_state
from app.observability.demo_logger import DemoLogger
from app.observability.langsmith import (
    get_current_public_run_url,
    get_current_run_url,
    get_latest_public_run_url,
    get_latest_run_url,
    run_with_root_trace,
)
from app.observability.token_usage import reset_token_usage, snapshot_token_usage
from app.schemas.state_schema import GraphState
from app.utils.target_scope import canonical_target_scope, exact_target_matches
from run_multiagent import NODE_DESCRIPTIONS, log_node_start
from visualize_json_logs import (
    extract_module_usage_details,
    extract_node_runs,
    extract_token_usage_by_module,
    filter_runs_by_scope,
    load_events,
    plot_node_latency_bar,
    plot_token_io_and_prompt,
    plot_token_usage_by_module,
    plot_waterfall,
    write_summary_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MultiAgent workflow from a natural-language task."
    )
    parser.add_argument("--task", required=True, help="Natural-language task description.")
    parser.add_argument("--dataset", required=True, help="Dataset directory or CSV file.")
    parser.add_argument(
        "--target",
        default="",
        help="Optional target column or natural-language target hint.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the environment file containing LLM configuration.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="LangGraph recursion limit for the workflow run.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip human confirmation after Reader and Background Reviewer approve.",
    )
    parser.add_argument(
        "--demo-log-file",
        default="",
        help=(
            "Path to local JSONL event log. If omitted, a timestamped log is "
            "created under artifacts/logs/ using dataset name."
        ),
    )
    return parser.parse_args(argv)


def configure_runtime(env_file: str) -> None:
    load_dotenv(env_file)
    mpl_cache = Path(os.environ.get("MPLCONFIGDIR", "/tmp/multiagent-mpl-cache"))
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    warnings.filterwarnings("ignore", category=FutureWarning)


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_for_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    return cleaned or "dataset"


def _dataset_name_for_log(raw_dataset_arg: str) -> str:
    candidate = Path(raw_dataset_arg).stem or Path(raw_dataset_arg).name
    return _sanitize_for_filename(candidate)


def resolve_demo_log_path(
    dataset_arg: str,
    cli_value: str,
    artifact_root: str | None = None,
) -> Path:
    if cli_value.strip():
        return Path(cli_value).expanduser()
    if artifact_root:
        run_label = Path(artifact_root).name
        # Prefer a file stem that matches "<dataset>_<YYYYMMDD_HHMMSS>" by
        # stripping the trailing short run id suffix "_<8hex>" if present.
        stem = run_label
        if len(run_label) > 9 and run_label[-9:].startswith("_") and all(ch in "0123456789abcdef" for ch in run_label[-8:]):
            stem = run_label[:-9]
        log_dir = Path(artifact_root) / "logs"
        return log_dir / f"{stem}.jsonl"
    filename = f"{_dataset_name_for_log(dataset_arg)}_{_timestamp_for_filename()}.jsonl"
    return Path("artifacts/logs") / filename


def _trace_metadata(state: GraphState, *, phase: str, local_log_path: str) -> dict[str, Any]:
    return {
        "workflow": "main_cli",
        "phase": phase,
        "thread_id": state.runtime_state.thread_id or "",
        "run_id": state.runtime_state.run_id or state.task_config.task_id or "",
        "dataset": _dataset_name_for_log(state.task_config.dataset_root or state.task_config.dataset_paths.train or "dataset"),
        "task_type": state.target_spec.task_family or state.problem_understanding.predicted_task_type or "",
        "target_column": state.task_config.target_column or state.problem_understanding.target_column or "",
        "local_log_path": local_log_path,
        "token_usage_snapshot": snapshot_token_usage(),
    }


def _event_key(payload: dict[str, Any], node_name: str) -> str:
    for key in ("id", "task_id", "task", "name"):
        value = payload.get(key)
        if value:
            return str(value)
    return node_name


async def prepare_confirmed_state(
    state: GraphState,
    *,
    auto_yes: bool,
    demo_logger: DemoLogger,
) -> GraphState | None:
    """Run Reader + Background Reviewer until approved, fatal, or user exits."""

    for _ in range(3):
        reader_event_key = f"reader_{int(time.time() * 1000)}"
        demo_logger.log_node_start(node_name="reader", event_key=reader_event_key, dataset=_dataset_name_for_log(state.task_config.dataset_root))
        state = _merge_state(state, await run_reader(state))
        demo_logger.log_node_done(
            node_name="reader",
            event_key=reader_event_key,
            dataset=_dataset_name_for_log(state.task_config.dataset_root),
            current_phase="reader",
            token_usage=snapshot_token_usage().get("reader", {}),
            token_usage_by_module=snapshot_token_usage(),
        )

        reviewer_event_key = f"background_reviewer_{int(time.time() * 1000)}"
        demo_logger.log_node_start(
            node_name="background_reviewer",
            event_key=reviewer_event_key,
            dataset=_dataset_name_for_log(state.task_config.dataset_root),
        )
        state = _merge_state(state, await run_bg_reviewer(state))
        demo_logger.log_node_done(
            node_name="background_reviewer",
            event_key=reviewer_event_key,
            dataset=_dataset_name_for_log(state.task_config.dataset_root),
            current_phase="background_reviewer",
            token_usage=snapshot_token_usage().get("background_reviewer", {}),
            token_usage_by_module=snapshot_token_usage(),
        )
        _print_reader_understanding(state)

        status = state.review_state.latest_review_status
        if status == "approved":
            if auto_yes or _confirm("Continue the full workflow to the business report? [y/N] "):
                return state
            print("User did not confirm; exiting.")
            return None

        if status == "fatal":
            errors = "; ".join(str(item) for item in state.review_state.fatal_errors)
            raise RuntimeError(errors or state.review_state.last_review_comment)

        if _target_retry_needed(state):
            state = _merge_target_hint(state, _prompt_for_target(state))
            continue

        raise RuntimeError(state.review_state.last_review_comment or "Reader review failed.")

    raise RuntimeError("Reader review did not converge after target clarification attempts.")


async def run_full_workflow(
    state: GraphState,
    *,
    recursion_limit: int,
    demo_logger: DemoLogger,
) -> dict[str, str]:
    from app.graph.workflow import build_workflow

    graph = build_workflow(stop_after_technical_summary=False)
    config = {
        "recursion_limit": recursion_limit,
        "configurable": {"thread_id": state.runtime_state.thread_id},
    }
    artifact_paths = _artifact_paths(state)

    print("Starting full MultiAgent LangGraph workflow.", flush=True)
    async for event in graph.astream(state, stream_mode="debug", config=config):
        payload = event.get("payload") or {}
        node_name = str(payload.get("name") or "")
        event_key = _event_key(payload, node_name)
        if event.get("type") == "task" and node_name:
            log_node_start(node_name)
            demo_logger.log_node_start(
                node_name=node_name,
                event_key=event_key,
                dataset=_dataset_name_for_log(state.task_config.dataset_root),
            )
            continue
        if event.get("type") != "task_result" or not node_name:
            continue
        if payload.get("error"):
            demo_logger.log_node_error(
                node_name=node_name,
                event_key=event_key,
                dataset=_dataset_name_for_log(state.task_config.dataset_root),
                error=payload["error"],
            )
            raise RuntimeError(f"{node_name} failed: {payload['error']}")

        result = payload.get("result")
        if isinstance(result, dict):
            _log_node_result(node_name, result)
            artifact_paths.update(_paths_from_update(result))
            demo_logger.log_node_done(
                node_name=node_name,
                event_key=event_key,
                dataset=_dataset_name_for_log(state.task_config.dataset_root),
                current_phase=str(result.get("current_phase") or ""),
                technical_report_path=str((result.get("report_state") or {}).get("technical_report_path") or ""),
                token_usage=snapshot_token_usage().get(node_name, {}),
                token_usage_by_module=snapshot_token_usage(),
            )
        else:
            print(f"[DONE] {node_name} completed.", flush=True)
            demo_logger.log_node_done(
                node_name=node_name,
                event_key=event_key,
                dataset=_dataset_name_for_log(state.task_config.dataset_root),
                token_usage=snapshot_token_usage().get(node_name, {}),
                token_usage_by_module=snapshot_token_usage(),
            )

    return artifact_paths


def _merge_state(state: GraphState, update: dict[str, Any]) -> GraphState:
    data = state.model_dump(mode="python")
    for key, value in update.items():
        if key in {"phase_history", "messages"} and isinstance(value, list):
            data[key] = list(data.get(key) or []) + value
        elif isinstance(data.get(key), dict) and isinstance(value, dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return GraphState(**data)


def _merge_target_hint(state: GraphState, target_hint: str) -> GraphState:
    data = state.model_dump(mode="python")
    data["task_config"]["target_hint"] = target_hint
    columns = _train_columns(state)
    target_column, target_columns = canonical_target_scope(
        target_columns=exact_target_matches(target_hint, columns)
    )
    data["task_config"]["target_column"] = target_column
    data["task_config"]["target_columns"] = target_columns
    return GraphState(**data)


def _print_reader_understanding(state: GraphState) -> None:
    problem = state.problem_understanding
    paths = state.task_config.dataset_paths
    risks = list(problem.risks or [])
    comment = state.review_state.last_review_comment
    target_column, target_columns = canonical_target_scope(
        target_columns=state.target_spec.target_columns
    )

    print("\nTask understanding:")
    print(f"- Task summary: {problem.task_summary or '(not generated)'}")
    print(
        f"- Task type: "
        f"{state.target_spec.task_family or problem.predicted_task_type or '(undetermined)'}"
    )
    if len(target_columns) > 1:
        print(f"- Target columns: {', '.join(target_columns)}")
    else:
        print(f"- Target column: {target_column or '(undetermined)'}")
    print(f"- Train data: {paths.train or '(not found)'}")
    print(f"- Test data: {paths.test or '(not found)'}")
    print(f"- Sample submission: {paths.submission or '(not found)'}")
    print(f"- Candidate metrics: {', '.join(problem.candidate_metrics) or '(not generated)'}")
    if risks:
        print("- Risks/warnings:")
        for risk in risks:
            print(f"  - {risk}")
    if comment:
        print(f"- Reviewer: {comment}")


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() in {"y", "yes", "confirm"}


def _target_retry_needed(state: GraphState) -> bool:
    if state.review_state.latest_review_status != "retry":
        return False
    if state.target_spec.status != "resolved":
        return True
    failures = " ".join(state.review_state.review_history[-1].get("blocking_failures", []))
    return any(
        marker in failures.lower()
        for marker in ("target column", "target columns", "target scope", "target unresolved")
    )


def _prompt_for_target(state: GraphState) -> str:
    columns = _train_columns(state)
    print("\nReader could not determine the target scope.")
    if columns:
        print(f"Candidate columns: {', '.join(columns)}")
    target = input(
        "Describe what you want to predict, or enter the exact target column name(s): "
    ).strip()
    if not target:
        raise RuntimeError("No target clarification was provided; cannot continue.")
    return target


def _train_columns(state: GraphState) -> list[str]:
    if not state.review_state.review_history:
        return []
    details = state.review_state.review_history[-1].get("details") or {}
    columns = details.get("train_columns")
    return [str(column) for column in columns] if isinstance(columns, list) else []


def _log_node_result(node_name: str, result: dict[str, Any]) -> None:
    parts: list[str] = []
    current_phase = result.get("current_phase")
    if current_phase:
        parts.append(f"phase={current_phase}")
    review_state = result.get("review_state")
    if isinstance(review_state, dict) and review_state.get("latest_review_status"):
        parts.append(f"review={review_state['latest_review_status']}")
    report_state = result.get("report_state")
    if isinstance(report_state, dict):
        for key in ("technical_report_path", "business_report_path"):
            if report_state.get(key):
                parts.append(f"{key}={report_state[key]}")
    suffix = f" [{', '.join(parts)}]" if parts else ""
    description = NODE_DESCRIPTIONS.get(node_name, "LangGraph node")
    print(f"[DONE] {node_name} completed.{suffix} - {description}", flush=True)


def _artifact_paths(state: GraphState) -> dict[str, str]:
    return {}


def _paths_from_update(update: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    report_state = update.get("report_state")
    if isinstance(report_state, dict):
        for key in ("technical_report_path", "business_report_path"):
            if report_state.get(key):
                paths[key] = str(report_state[key])
    artifacts = update.get("artifacts")
    if isinstance(artifacts, dict):
        for key in (
            "business_report",
            "technical_report",
            "prediction_file",
            "model_results",
            "best_model",
            "feature_summary",
            "cleaning_report",
        ):
            if artifacts.get(key) and _path_exists(str(artifacts[key])):
                paths[key] = str(artifacts[key])
    return paths


def _path_exists(raw_path: str) -> bool:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.exists()


def _path_identity(raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve(strict=False)).casefold()


def _print_final_paths(paths: dict[str, str]) -> None:
    print("\nFinal artifact paths:")
    printed_paths: set[str] = set()
    for key in (
        "business_report",
        "business_report_path",
        "technical_report",
        "technical_report_path",
        "prediction_file",
        "model_results",
        "best_model",
        "feature_summary",
        "cleaning_report",
    ):
        value = paths.get(key)
        if not value:
            continue
        path_key = _path_identity(value)
        if path_key in printed_paths:
            continue
        printed_paths.add(path_key)
        print(f"- {key}: {value}")


def generate_log_visualizations(log_path: Path, *, output_dir: Path | None = None) -> dict[str, Path]:
    """Generate summary + waterfall + token charts for one run log."""
    events = load_events(log_path)
    runs = filter_runs_by_scope(events, extract_node_runs(events), scope="latest")
    if not runs:
        return {}

    output_dir = (output_dir or (log_path.parent.parent / "figures")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = log_path.stem
    summary_csv = output_dir / f"{stem}_summary.csv"
    node_latency_png = output_dir / f"{stem}_node_latency.png"
    waterfall_png = output_dir / f"{stem}_waterfall.png"
    token_usage_png = output_dir / f"{stem}_token_usage.png"
    token_io_prompt_png = output_dir / f"{stem}_token_io_prompt.png"

    write_summary_csv(runs, summary_csv)
    plot_node_latency_bar(runs, node_latency_png)
    plot_waterfall(runs, waterfall_png)

    token_usage = extract_token_usage_by_module(events, scope="latest")
    if token_usage:
        plot_token_usage_by_module(token_usage, token_usage_png)

    module_usage = extract_module_usage_details(events, scope="latest")
    if module_usage:
        plot_token_io_and_prompt(module_usage, token_io_prompt_png)

    outputs = {
        "summary_csv": summary_csv,
        "node_latency_png": node_latency_png,
        "waterfall_png": waterfall_png,
    }
    if token_usage:
        outputs["token_usage_png"] = token_usage_png
    if module_usage:
        outputs["token_io_prompt_png"] = token_io_prompt_png
    return outputs


def write_run_summary(
    *,
    state: GraphState,
    dataset_name: str,
    paths: dict[str, str],
    trace_url: str,
) -> None:
    artifact_root = Path(state.runtime_state.artifact_root)
    metadata_dir = artifact_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": state.runtime_state.run_id,
        "thread_id": state.runtime_state.thread_id,
        "dataset": dataset_name,
        "artifact_root": str(artifact_root),
        "trace_url": trace_url,
        "business_report_path": paths.get("business_report_path") or paths.get("business_report") or "",
        "technical_report_path": paths.get("technical_report_path") or paths.get("technical_report") or "",
        "prediction_file": paths.get("prediction_file") or "",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (metadata_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_index_path = Path("artifacts") / "run_index.jsonl"
    run_index_path.parent.mkdir(parents=True, exist_ok=True)
    with run_index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    demo_logger: DemoLogger | None = None
    try:
        configure_runtime(args.env_file)
        reset_token_usage()
        state = build_initial_state(
            task=args.task,
            dataset=args.dataset,
            target=args.target,
            workspace_root=".",
        )
        dataset_name = _dataset_name_for_log(args.dataset)
        demo_log_path = resolve_demo_log_path(
            args.dataset,
            args.demo_log_file,
            artifact_root=state.runtime_state.artifact_root,
        )
        demo_logger = DemoLogger(demo_log_path)
        print(f"Demo JSONL log: {demo_logger.path}", flush=True)
        demo_logger.log_workflow_start(
            dataset=dataset_name,
            thread_id=state.runtime_state.thread_id,
            recursion_limit=args.recursion_limit,
        )

        async def _run_pipeline_under_one_trace() -> tuple[GraphState | None, dict[str, str], str]:
            confirmed = await prepare_confirmed_state(state, auto_yes=args.yes, demo_logger=demo_logger)
            if confirmed is None:
                return None, {}, get_current_run_url()
            final_paths = await run_full_workflow(
                confirmed,
                recursion_limit=args.recursion_limit,
                demo_logger=demo_logger,
            )
            return confirmed, final_paths, get_current_run_url()

        confirmed_state, paths, trace_url = await run_with_root_trace(
            run_name=demo_logger.path.stem,
            tags=["main-cli", "workflow"],
            metadata=_trace_metadata(state, phase="full_workflow", local_log_path=str(demo_logger.path)),
            coro_fn=_run_pipeline_under_one_trace,
        )
        if confirmed_state is None:
            demo_logger.log_workflow_end(
                dataset=dataset_name,
                report_path="",
                token_usage_by_module=snapshot_token_usage(),
            )
            return 0
        public_trace_url = get_current_public_run_url()
        if not public_trace_url:
            public_trace_url = get_latest_public_run_url(
                thread_id=confirmed_state.runtime_state.thread_id,
                dataset=_dataset_name_for_log(args.dataset),
            )
        if not trace_url:
            trace_url = get_latest_run_url(
                thread_id=confirmed_state.runtime_state.thread_id,
                dataset=_dataset_name_for_log(args.dataset),
            )
        _print_final_paths(paths)
        if public_trace_url:
            print(f"\nLangSmith public trace: {public_trace_url}")
        elif trace_url:
            print(f"\nLangSmith trace: {trace_url}")
        write_run_summary(
            state=confirmed_state,
            dataset_name=dataset_name,
            paths=paths,
            trace_url=public_trace_url or trace_url,
        )
        demo_logger.log_workflow_end(
            dataset=dataset_name,
            report_path=paths.get("business_report_path") or paths.get("technical_report_path") or "",
            token_usage_by_module=snapshot_token_usage(),
        )
        visual_outputs = generate_log_visualizations(
            demo_logger.path,
            output_dir=Path(confirmed_state.runtime_state.artifact_root) / "figures",
        )
        if visual_outputs:
            print("\nLog visualizations:")
            for label, output_path in visual_outputs.items():
                print(f"- {label}: {output_path}")
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.", file=sys.stderr)
        if demo_logger is not None:
            demo_logger.log_workflow_error(
                dataset=_dataset_name_for_log(args.dataset),
                error=KeyboardInterrupt("Interrupted by user"),
                traceback_text="",
            )
        return 130
    except Exception as exc:
        print(f"\nWorkflow failed: {exc}", file=sys.stderr)
        if demo_logger is not None:
            demo_logger.log_workflow_error(
                dataset=_dataset_name_for_log(args.dataset),
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            fallback_output_dir = demo_logger.path.parent.parent / "figures"
            if "confirmed_state" in locals() and confirmed_state is not None:
                fallback_output_dir = Path(confirmed_state.runtime_state.artifact_root) / "figures"
            elif "state" in locals() and state is not None:
                fallback_output_dir = Path(state.runtime_state.artifact_root) / "figures"
            visual_outputs = generate_log_visualizations(
                demo_logger.path,
                output_dir=fallback_output_dir,
            )
            if visual_outputs:
                print("\nLog visualizations:")
                for label, output_path in visual_outputs.items():
                    print(f"- {label}: {output_path}")
        return 1
    finally:
        if demo_logger is not None:
            demo_logger.close()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
