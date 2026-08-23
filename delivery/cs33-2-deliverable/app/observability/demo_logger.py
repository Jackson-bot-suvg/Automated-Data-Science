"""Local demo logger for real-time node events and replay."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_iso_now() -> str:
    return datetime.now(UTC).isoformat()


class DemoLogger:
    """Append structured workflow events to a local JSONL file."""

    def __init__(self, log_path: str | Path, *, flush_every_write: bool = True) -> None:
        self._path = Path(log_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("a", encoding="utf-8")
        self._flush_every_write = flush_every_write
        self._workflow_start_ms = int(time.time() * 1000)
        self._task_start_times_ms: dict[str, int] = {}

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def log_workflow_start(self, *, dataset: str, thread_id: str, recursion_limit: int) -> None:
        self._write(
            {
                "event": "workflow_start",
                "timestamp": _utc_iso_now(),
                "dataset": dataset,
                "thread_id": thread_id,
                "recursion_limit": recursion_limit,
            }
        )

    def log_workflow_end(
        self,
        *,
        dataset: str,
        report_path: str,
        token_usage_by_module: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        self._write(
            {
                "event": "workflow_end",
                "timestamp": _utc_iso_now(),
                "dataset": dataset,
                "report_path": report_path,
                "latency_ms": now_ms - self._workflow_start_ms,
                "token_usage_by_module": token_usage_by_module or {},
            }
        )

    def log_node_start(self, *, node_name: str, event_key: str, dataset: str) -> None:
        self._task_start_times_ms[event_key] = int(time.time() * 1000)
        self._write(
            {
                "event": "node_start",
                "timestamp": _utc_iso_now(),
                "dataset": dataset,
                "node_name": node_name,
                "event_key": event_key,
                "status": "running",
            }
        )

    def log_node_done(
        self,
        *,
        node_name: str,
        event_key: str,
        dataset: str,
        current_phase: str = "",
        review_status: str = "",
        pending_review_phase: str = "",
        technical_report_path: str = "",
        token_usage: dict[str, Any] | None = None,
        token_usage_by_module: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        started_ms = self._task_start_times_ms.pop(event_key, None)
        self._write(
            {
                "event": "node_result",
                "timestamp": _utc_iso_now(),
                "dataset": dataset,
                "node_name": node_name,
                "event_key": event_key,
                "status": "done",
                "latency_ms": (now_ms - started_ms) if started_ms is not None else None,
                "current_phase": current_phase,
                "review_status": review_status,
                "pending_review_phase": pending_review_phase,
                "technical_report_path": technical_report_path,
                "token_usage": token_usage or {},
                "token_usage_by_module": token_usage_by_module or {},
            }
        )

    def log_node_error(self, *, node_name: str, event_key: str, dataset: str, error: Any) -> None:
        now_ms = int(time.time() * 1000)
        started_ms = self._task_start_times_ms.pop(event_key, None)
        self._write(
            {
                "event": "node_result",
                "timestamp": _utc_iso_now(),
                "dataset": dataset,
                "node_name": node_name,
                "event_key": event_key,
                "status": "error",
                "latency_ms": (now_ms - started_ms) if started_ms is not None else None,
                "error": self._stringify_error(error),
            }
        )

    def log_workflow_error(self, *, dataset: str, error: Any, traceback_text: str = "") -> None:
        self._write(
            {
                "event": "workflow_error",
                "timestamp": _utc_iso_now(),
                "dataset": dataset,
                "status": "error",
                "error": self._stringify_error(error),
                "traceback": traceback_text,
            }
        )

    def _write(self, payload: dict[str, Any]) -> None:
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if self._flush_every_write:
            self._fp.flush()

    @staticmethod
    def _stringify_error(error: Any) -> str:
        if isinstance(error, BaseException):
            return f"{error.__class__.__name__}: {error}"
        return str(error)

