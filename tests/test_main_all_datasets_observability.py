"""Run main-style full workflow across datasets with observability outputs.

This script executes the full graph (through business report) sequentially for
multiple datasets, writes local demo JSONL logs, and attempts to pull LangSmith
run summaries for each dataset by thread_id.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.intake import build_initial_state
from app.observability.demo_logger import DemoLogger
from app.observability.langsmith import get_langsmith_config
from app.observability.token_usage import reset_token_usage, snapshot_token_usage
from app.validation_matrix import PRIMARY_ACCEPTANCE_DATASETS, SECONDARY_SMOKE_DATASETS
from main import configure_runtime, prepare_confirmed_state
from run_multiagent import event_key_from_payload, extract_node_observability_fields, log_node_result


DATASETS: list[dict[str, str]] = [
    {
        "name": spec.key,
        "task": spec.task,
        "dataset": spec.dataset,
        "target": spec.target,
        "primary_acceptance": spec.primary_acceptance,
    }
    for spec in (*PRIMARY_ACCEPTANCE_DATASETS, *SECONDARY_SMOKE_DATASETS)
]


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_for_filename(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    return out or "dataset"


def _resolve_demo_log_path(cli_value: str) -> Path:
    if cli_value.strip():
        return Path(cli_value).expanduser()
    return Path("artifacts/logs") / f"main_all_datasets_{_timestamp_for_filename()}.jsonl"


def _result(name: str, status: str, elapsed: float, report_paths: dict[str, str], error: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "report_paths": report_paths,
        "error": error,
    }


def _update_paths_from_result(paths: dict[str, str], result: dict[str, Any]) -> None:
    report_state = result.get("report_state")
    if isinstance(report_state, dict):
        for key in ("technical_report_path", "business_report_path"):
            if report_state.get(key):
                paths[key] = str(report_state[key])
    artifacts = result.get("artifacts")
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
            if artifacts.get(key):
                paths[key] = str(artifacts[key])


def _fetch_langsmith_summary(thread_id: str) -> dict[str, Any]:
    cfg = get_langsmith_config()
    if not cfg.enabled:
        return {"enabled": False, "reason": cfg.reason or "LangSmith disabled."}
    try:
        import langsmith
    except ImportError:
        return {"enabled": False, "reason": "langsmith package not installed."}

    try:
        client = langsmith.Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
        kwargs: dict[str, Any] = {"filter": f'eq(metadata_key, "thread_id", "{thread_id}")'}
        if cfg.project_name:
            kwargs["project_name"] = cfg.project_name
        runs = list(client.list_runs(**kwargs, limit=50))
        total_tokens = 0
        for run in runs:
            usage = getattr(run, "usage_metadata", None) or {}
            if isinstance(usage, dict):
                total_tokens += int(usage.get("total_tokens") or 0)
        return {
            "enabled": True,
            "project": cfg.project_name or "",
            "thread_id": thread_id,
            "run_count": len(runs),
            "total_tokens": total_tokens,
            "run_ids": [str(getattr(run, "id", "")) for run in runs if getattr(run, "id", None)],
        }
    except Exception as exc:
        return {
            "enabled": True,
            "thread_id": thread_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _run_one_dataset(ds: dict[str, str], recursion_limit: int, demo_logger: DemoLogger) -> dict[str, Any]:
    from app.graph.workflow import build_workflow

    name = ds["name"]
    thread_id = f"{_sanitize_for_filename(name)}-main-test-{_timestamp_for_filename()}"
    start = time.time()
    report_paths: dict[str, str] = {}
    reviewer_retries: list[dict[str, Any]] = []

    try:
        reset_token_usage()
        demo_logger.log_workflow_start(dataset=name, thread_id=thread_id, recursion_limit=recursion_limit)
        state = build_initial_state(
            task=ds["task"],
            dataset=ds["dataset"],
            target=ds["target"],
            workspace_root=".",
            thread_id=thread_id,
        )
        confirmed_state = await prepare_confirmed_state(state, auto_yes=True)
        if confirmed_state is None:
            raise RuntimeError("Workflow confirmation rejected.")

        graph = build_workflow(stop_after_technical_summary=False)
        config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": thread_id}}
        async for event in graph.astream(confirmed_state, stream_mode="debug", config=config):
            event_type = event.get("type")
            payload = event.get("payload") or {}
            node_name = str(payload.get("name") or "")
            if not node_name:
                continue
            event_key = event_key_from_payload(payload, node_name)

            if event_type == "task":
                demo_logger.log_node_start(node_name=node_name, event_key=event_key, dataset=name)
                continue
            if event_type != "task_result":
                continue
            if payload.get("error"):
                demo_logger.log_node_error(
                    node_name=node_name,
                    event_key=event_key,
                    dataset=name,
                    error=payload.get("error"),
                )
                raise RuntimeError(f"{node_name} failed: {payload['error']}")

            result = payload.get("result")
            fields = extract_node_observability_fields(result)
            token_snapshot = snapshot_token_usage()
            demo_logger.log_node_done(
                node_name=node_name,
                event_key=event_key,
                dataset=name,
                current_phase=fields["current_phase"],
                review_status=fields["review_status"],
                pending_review_phase=fields["pending_review_phase"],
                technical_report_path=fields["technical_report_path"],
                token_usage=token_snapshot.get(node_name, {}),
            )
            if isinstance(result, dict):
                _ = log_node_result(node_name, result)

                _update_paths_from_result(report_paths, result)
                if node_name == "reviewer":
                    review_state = result.get("review_state")
                    if isinstance(review_state, dict) and review_state.get("latest_review_status") == "retry":
                        history = review_state.get("review_history") or []
                        latest = history[-1] if isinstance(history, list) and history else {}
                        blocking = latest.get("blocking_failures") if isinstance(latest, dict) else []
                        comment = review_state.get("last_review_comment") or ""
                        retry_record = {
                            "pending_phase": review_state.get("pending_review_phase") or "",
                            "comment": str(comment),
                            "blocking_failures": list(blocking) if isinstance(blocking, list) else [],
                        }
                        reviewer_retries.append(retry_record)
                        print(
                            f"  -> reviewer retry ({retry_record['pending_phase']}): {retry_record['comment']}",
                            flush=True,
                        )
                        if retry_record["blocking_failures"]:
                            for item in retry_record["blocking_failures"]:
                                print(f"     - blocking: {item}", flush=True)

        demo_logger.log_workflow_end(
            dataset=name,
            report_path=report_paths.get("business_report_path", ""),
            token_usage_by_module=snapshot_token_usage(),
        )
        langsmith_summary = _fetch_langsmith_summary(thread_id)
        return {
            **_result(name, "PASS", time.time() - start, report_paths),
            "thread_id": thread_id,
            "langsmith": langsmith_summary,
            "reviewer_retries": reviewer_retries,
        }
    except Exception as exc:
        demo_logger.log_workflow_error(dataset=name, error=exc, traceback_text=traceback.format_exc())
        return {
            **_result(name, "FAIL", time.time() - start, report_paths, f"{type(exc).__name__}: {exc}"),
            "thread_id": thread_id,
            "langsmith": _fetch_langsmith_summary(thread_id),
            "reviewer_retries": reviewer_retries,
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run full main-style workflow across datasets with observability.")
    parser.add_argument("--only", type=str, default="", help="Only run one dataset by name.")
    parser.add_argument("--limit", type=int, default=0, help="Run first N datasets.")
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include secondary smoke datasets; default runs only the 4 representative acceptance tasks.",
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Environment file path.")
    parser.add_argument("--recursion-limit", type=int, default=100, help="LangGraph recursion limit.")
    parser.add_argument("--demo-log-file", type=str, default="", help="Optional demo JSONL output path.")
    args = parser.parse_args()

    load_dotenv(args.env_file)
    configure_runtime(args.env_file)

    datasets = DATASETS if args.include_smoke else [ds for ds in DATASETS if ds["primary_acceptance"]]
    if args.only:
        datasets = [ds for ds in datasets if ds["name"] == args.only]
        if not datasets:
            raise SystemExit(f"Dataset not found: {args.only}")
    if args.limit > 0:
        datasets = datasets[: args.limit]

    demo_path = _resolve_demo_log_path(args.demo_log_file)
    demo_logger = DemoLogger(demo_path)
    print(f"Demo JSONL log: {demo_logger.path}", flush=True)

    results: list[dict[str, Any]] = []
    try:
        for idx, ds in enumerate(datasets, 1):
            print(f"[{idx}/{len(datasets)}] Running {ds['name']}...", flush=True)
            result = await _run_one_dataset(ds, args.recursion_limit, demo_logger)
            results.append(result)
            print(f"  -> {result['status']} ({result['elapsed_s']}s)", flush=True)
            ls = result.get("langsmith", {})
            if isinstance(ls, dict):
                if ls.get("run_count") is not None:
                    print(
                        f"  -> LangSmith runs: {ls.get('run_count', 0)}, tokens: {ls.get('total_tokens', 0)}",
                        flush=True,
                    )
                elif ls.get("reason"):
                    print(f"  -> LangSmith skipped: {ls['reason']}", flush=True)
                elif ls.get("error"):
                    print(f"  -> LangSmith pull error: {ls['error']}", flush=True)
    finally:
        demo_logger.close()

    report_path = Path("artifacts/test_main_all_datasets_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    print(f"Finished: {passed} PASS / {failed} FAIL / {len(results)} TOTAL", flush=True)
    print(f"Report: {report_path}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
