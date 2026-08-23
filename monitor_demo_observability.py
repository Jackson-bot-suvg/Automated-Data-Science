"""Live monitor for local demo logs + LangSmith run status.

Usage example:
    python monitor_demo_observability.py --project 26s1-cs33-2-dev --thread-id titanic-cli-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for lightweight environments
    load_dotenv = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch local demo JSONL and poll LangSmith runs in parallel."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file for LANGSMITH_* values.",
    )
    parser.add_argument(
        "--json-log-file",
        default="",
        help=(
            "Local demo JSONL log path. If omitted, uses the most recent "
            "artifacts/logs/<dataset>_<timestamp>.jsonl."
        ),
    )
    parser.add_argument(
        "--project",
        default="",
        help="LangSmith project name. Defaults to LANGSMITH_PROJECT in env.",
    )
    parser.add_argument(
        "--thread-id",
        default="",
        help="Optional thread id filter for both local and LangSmith events.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset name for default log naming (e.g. titanic).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="LangSmith polling interval in seconds.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=50,
        help="Max runs fetched per poll request.",
    )
    parser.add_argument(
        "--ls-log-file",
        default="",
        help=(
            "Local JSONL path to persist polled LangSmith events. "
            "If omitted, uses artifacts/logs/<dataset>_<timestamp>_langsmith.jsonl."
        ),
    )
    return parser.parse_args()


def format_event(prefix: str, payload: dict[str, Any]) -> str:
    ts = str(payload.get("timestamp") or payload.get("start_time") or "-")
    event = str(payload.get("event") or payload.get("run_type") or "run")
    status = str(payload.get("status") or payload.get("state") or "-")
    name = str(payload.get("node_name") or payload.get("name") or payload.get("run_name") or "-")
    latency = payload.get("latency_ms")
    latency_text = f" latency={latency}ms" if latency is not None else ""
    total_tokens = payload.get("total_tokens")
    token_text = f" tokens={total_tokens}" if total_tokens not in (None, "") else ""
    total_cost = payload.get("total_cost")
    cost_text = f" cost={total_cost}" if total_cost not in (None, "") else ""
    return f"{prefix} {ts} event={event} status={status} name={name}{latency_text}{token_text}{cost_text}"


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_for_filename(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    return sanitized or "dataset"


def _derive_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset.strip():
        raw = args.dataset.strip()
        p = Path(raw)
        # Align with main.py's filename logic: use directory/file stem.
        return p.stem or p.name or raw
    if args.thread_id.strip():
        raw = args.thread_id.strip()
        if raw.endswith("-cli-run"):
            return raw.removesuffix("-cli-run")
        return raw
    return "dataset"


def resolve_local_demo_log_path(args: argparse.Namespace) -> Path:
    if args.json_log_file.strip():
        return Path(args.json_log_file).expanduser()

    logs_dir = Path("artifacts/logs")
    dataset_prefix = _sanitize_for_filename(_derive_dataset_name(args))
    pattern = f"{dataset_prefix}_*.jsonl"
    candidates = sorted(
        [path for path in logs_dir.glob(pattern) if not path.name.endswith("_langsmith.jsonl")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    # New per-run logs location: artifacts/runs/<run_label>/logs/*.jsonl
    runs_logs_dir = Path("artifacts/runs")
    if runs_logs_dir.exists():
        run_candidates = sorted(
            [
                p
                for p in runs_logs_dir.glob(f"*/logs/{dataset_prefix}_*.jsonl")
                if not p.name.endswith("_langsmith.jsonl")
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if run_candidates:
            return run_candidates[0]

    return logs_dir / f"{dataset_prefix}_{_timestamp_for_filename()}.jsonl"


def resolve_langsmith_log_path(args: argparse.Namespace) -> Path:
    if args.ls_log_file.strip():
        return Path(args.ls_log_file).expanduser()
    dataset_prefix = _sanitize_for_filename(_derive_dataset_name(args))
    return Path("artifacts/logs") / f"{dataset_prefix}_{_timestamp_for_filename()}_langsmith.jsonl"


class LocalJsonlTail:
    def __init__(self, path: str | Path, thread_id_filter: str = "") -> None:
        self._path = Path(path).expanduser()
        self._thread_id_filter = thread_id_filter.strip()
        self._offset = 0

    def read_new_events(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as fp:
            fp.seek(self._offset)
            lines = fp.readlines()
            self._offset = fp.tell()

        events: list[dict[str, Any]] = []
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if self._thread_id_filter:
                payload_thread = str(payload.get("thread_id") or "")
                if payload_thread and payload_thread != self._thread_id_filter:
                    continue
            events.append(payload)
        return events


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()


class LangSmithPoller:
    def __init__(self, project: str, thread_id_filter: str, max_runs: int) -> None:
        self._project = project
        self._thread_id_filter = thread_id_filter.strip()
        self._max_runs = max_runs
        self._seen_status: dict[str, str] = {}
        self._client = self._build_client()

    @staticmethod
    def _build_client() -> Any:
        try:
            from langsmith import Client
        except ImportError as exc:
            raise RuntimeError(
                "langsmith package is not installed. Please install project dependencies first."
            ) from exc
        return Client()

    def poll(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        try:
            runs = self._client.list_runs(project_name=self._project, limit=self._max_runs)
        except Exception as exc:
            return [
                {
                    "event": "langsmith_poll_error",
                    "status": "error",
                    "name": "langsmith",
                    "error": str(exc),
                    "timestamp": "",
                }
            ]

        for run in runs:
            payload = self._normalize_run(run)
            if self._thread_id_filter and payload.get("thread_id") != self._thread_id_filter:
                continue
            run_id = payload.get("id", "")
            status = payload.get("status", "")
            previous = self._seen_status.get(run_id)
            if previous == status:
                continue
            self._seen_status[run_id] = status
            results.append(payload)
        return results

    def _normalize_run(self, run: Any) -> dict[str, Any]:
        name = str(getattr(run, "name", "") or "")
        run_type = str(getattr(run, "run_type", "") or "")
        run_id = str(getattr(run, "id", "") or "")
        start_time = getattr(run, "start_time", None)
        end_time = getattr(run, "end_time", None)
        status = self._derive_status(run)
        thread_id = self._extract_thread_id(run)
        latency_ms = None
        if start_time and end_time:
            try:
                latency_ms = int((end_time - start_time).total_seconds() * 1000)
            except Exception:
                latency_ms = None
        usage = self._extract_usage_and_cost(run)

        return {
            "event": "langsmith_run_update",
            "status": status,
            "name": name,
            "run_name": name,
            "run_type": run_type,
            "id": run_id,
            "thread_id": thread_id,
            "start_time": str(start_time or ""),
            "end_time": str(end_time or ""),
            "latency_ms": latency_ms,
            "error": str(getattr(run, "error", "") or ""),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "total_cost": usage["total_cost"],
        }

    @staticmethod
    def _derive_status(run: Any) -> str:
        error = getattr(run, "error", None)
        end_time = getattr(run, "end_time", None)
        if error:
            return "error"
        if end_time:
            return "completed"
        return "running"

    @staticmethod
    def _extract_thread_id(run: Any) -> str:
        extra = getattr(run, "extra", None)
        if isinstance(extra, dict):
            metadata = extra.get("metadata")
            if isinstance(metadata, dict):
                thread_id = metadata.get("thread_id")
                if thread_id:
                    return str(thread_id)
        metadata = getattr(run, "metadata", None)
        if isinstance(metadata, dict):
            thread_id = metadata.get("thread_id")
            if thread_id:
                return str(thread_id)
        return ""

    @staticmethod
    def _extract_usage_and_cost(run: Any) -> dict[str, Any]:
        input_tokens = None
        output_tokens = None
        total_tokens = None
        total_cost = None

        # Prefer explicit usage metadata when available.
        usage_metadata = getattr(run, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            input_tokens = usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens"))
            output_tokens = usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens"))
            total_tokens = usage_metadata.get("total_tokens")

        # Some SDK versions expose usage directly.
        usage = getattr(run, "usage", None)
        if isinstance(usage, dict):
            if input_tokens is None:
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
            if output_tokens is None:
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
            if total_tokens is None:
                total_tokens = usage.get("total_tokens")

        # Fallback: derive total when only prompt/completion are present.
        if total_tokens is None and isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
            total_tokens = int(input_tokens) + int(output_tokens)

        # Try cost from common fields.
        metadata = getattr(run, "metadata", None)
        if isinstance(metadata, dict):
            total_cost = metadata.get("total_cost", metadata.get("cost"))
        if total_cost is None and isinstance(usage_metadata, dict):
            total_cost = usage_metadata.get("total_cost", usage_metadata.get("cost"))

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }


def validate_langsmith_env(project_name: str) -> None:
    tracing = str(os.environ.get("LANGSMITH_TRACING", "")).lower()
    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    if not project_name:
        raise RuntimeError(
            "LangSmith project is empty. Set --project or LANGSMITH_PROJECT in env."
        )
    if not api_key:
        raise RuntimeError("LANGSMITH_API_KEY is missing in environment.")
    if tracing not in {"1", "true", "yes", "on"}:
        print(
            "[WARN] LANGSMITH_TRACING is not true. Polling may still work for existing data.",
            file=sys.stderr,
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv(args.env_file)
    project_name = args.project.strip() or str(os.environ.get("LANGSMITH_PROJECT", "")).strip()

    try:
        validate_langsmith_env(project_name)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1

    local_log_path = resolve_local_demo_log_path(args)
    local_tail = LocalJsonlTail(local_log_path, thread_id_filter=args.thread_id)
    try:
        poller = LangSmithPoller(
            project=project_name,
            thread_id_filter=args.thread_id,
            max_runs=args.max_runs,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1

    writer = JsonlWriter(resolve_langsmith_log_path(args))
    print(f"[INFO] Local JSON log: {local_log_path}", flush=True)
    print(f"[INFO] LangSmith JSON log: {writer.path}", flush=True)
    print(f"[INFO] LangSmith project: {project_name}", flush=True)
    if args.thread_id:
        print(f"[INFO] Thread filter: {args.thread_id}", flush=True)
    print("[INFO] Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            for payload in local_tail.read_new_events():
                print(format_event("[LOCAL]", payload), flush=True)
            for payload in poller.poll():
                print(format_event("[LS]", payload), flush=True)
                writer.write(
                    {
                        "source": "langsmith",
                        "polled_at": datetime.now(timezone.utc).isoformat(),
                        **payload,
                    }
                )
                error = payload.get("error")
                if error:
                    print(f"[LS] error={error}", flush=True)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n[INFO] Monitor stopped.", flush=True)
        return 0
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())

