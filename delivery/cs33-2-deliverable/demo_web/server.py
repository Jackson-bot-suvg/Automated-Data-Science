"""Local web demo server wired to the real main.py workflow.

Run from the repository root:

    python demo_web/server.py --port 8765
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import markdown


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent
RUN_ROOT = STATIC_ROOT / "runs"


TASKS: dict[str, dict[str, str]] = {
    "bank-churn": {
        "task": "Predict customer churn and translate model drivers into retention actions.",
        "dataset": "data/bank_churn",
        "target": "Exited",
    },
    "academic-success": {
        "task": "Classify student academic outcomes and summarize intervention opportunities.",
        "dataset": "data/classification_with_an_academic_success_dataset",
        "target": "Target",
    },
    "ghouls": {
        "task": "Classify compact tabular records with categorical and numeric morphology signals.",
        "dataset": "data/ghouls_goblins_and_ghosts_boo",
        "target": "type",
    },
    "house-prices": {
        "task": "Estimate property sale price and surface the highest-value property attributes.",
        "dataset": "data/house_prices",
        "target": "SalePrice",
    },
    "obesity-risks": {
        "task": "Classify obesity risk levels and translate health-pattern drivers into segment guidance.",
        "dataset": "data/obesity_risks",
        "target": "NObeyesdad",
    },
    "plate-defect": {
        "task": "Detect plate defect classes and summarize operational quality risks.",
        "dataset": "data/plate_defect",
        "target": "Pastry,Z_Scratch,K_Scatch,Stains,Dirtiness,Bumps,Other_Faults",
    },
    "spaceship": {
        "task": "Classify passenger transport outcome and explain cohort-level movement.",
        "dataset": "data/spaceship_titanic",
        "target": "Transported",
    },
    "titanic": {
        "task": "Predict passenger survival and produce an interpretable benchmark analysis.",
        "dataset": "data/titanic_autokaggle",
        "target": "Survived",
    },
}


class RunRecord:
    def __init__(self, run_id: str, task_id: str, log_path: Path, process: subprocess.Popen[str]) -> None:
        self.run_id = run_id
        self.task_id = task_id
        self.log_path = log_path
        self.process = process
        self.stdout_queue: queue.Queue[str] = queue.Queue()
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.started_ts = time.time()
        self.dataset_slug = Path(TASKS[task_id]["dataset"]).name
        self.artifact_root: Path | None = None
        self.last_snapshot_key = ""


runs: dict[str, RunRecord] = {}
runs_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the real workflow web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "CS33DemoHTTP/0.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - inherited name
        sys.stderr.write("[demo-web] " + format % args + "\n")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/demo-runs":
            self._handle_start_run()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/task-snapshot":
            self._handle_task_snapshot(parsed.query)
            return
        if parsed.path.startswith("/api/demo-runs/") and parsed.path.endswith("/events"):
            run_id = parsed.path.split("/")[3]
            self._handle_events(run_id)
            return
        if parsed.path == "/api/report":
            self._handle_report_file(parsed.query)
            return
        super().do_GET()

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object")
        return payload

    def _handle_start_run(self) -> None:
        try:
            payload = self._read_json_body()
            task_id = str(payload.get("taskSetId") or "").strip()
            task = TASKS.get(task_id)
            if not task:
                self._send_json({"error": f"Unknown taskSetId: {task_id}"}, HTTPStatus.BAD_REQUEST)
                return
            dataset_path = ROOT / task["dataset"]
            if not dataset_path.exists():
                self._send_json({"error": f"Dataset does not exist: {task['dataset']}"}, HTTPStatus.BAD_REQUEST)
                return
            run_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
            run_dir = RUN_ROOT / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "events.jsonl"
            process = self._spawn_workflow(task, log_path)
            record = RunRecord(run_id, task_id, log_path, process)
            with runs_lock:
                runs[run_id] = record
            threading.Thread(target=_drain_stdout, args=(record,), daemon=True).start()
            self._send_json(
                {
                    "runId": run_id,
                    "status": "running",
                    "taskSetId": task_id,
                    "logPath": str(log_path.relative_to(ROOT)),
                },
                HTTPStatus.CREATED,
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive demo boundary
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_task_snapshot(self, query: str) -> None:
        params = parse_qs(query)
        task_id = str(params.get("taskSetId", [""])[0] or "").strip()
        if task_id not in TASKS:
            self._send_json({"error": f"Unknown taskSetId: {task_id}"}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(_latest_snapshot_for_task(task_id) or {})

    def _spawn_workflow(self, task: dict[str, str], log_path: Path) -> subprocess.Popen[str]:
        python = _workflow_python()
        cmd = [
            str(python),
            "-u",
            "main.py",
            "--task",
            task["task"],
            "--dataset",
            task["dataset"],
            "--target",
            task["target"],
            "--yes",
            "--demo-log-file",
            str(log_path),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["PATH"] = f"{python.parent}{os.pathsep}{env.get('PATH', '')}"
        return subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def _handle_events(self, run_id: str) -> None:
        with runs_lock:
            record = runs.get(run_id)
        if not record:
            self.send_error(HTTPStatus.NOT_FOUND, "Run not found")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        self._send_sse(
            {
                "type": "run_attached",
                "runId": run_id,
                "taskSetId": record.task_id,
                "logPath": str(record.log_path.relative_to(ROOT)),
                "createdAt": record.created_at,
            }
        )

        offset = 0
        sent_process_end = False
        last_heartbeat = time.time()
        while True:
            lines, offset = _read_new_log_lines(record.log_path, offset)
            for line in lines:
                payload = _parse_json_line(line)
                if payload:
                    self._send_sse({"type": "workflow_event", "payload": payload})

            snapshot = _build_run_snapshot(record)
            if snapshot and snapshot["snapshotKey"] != record.last_snapshot_key:
                record.last_snapshot_key = str(snapshot["snapshotKey"])
                self._send_sse({"type": "run_snapshot", "payload": snapshot})

            while True:
                try:
                    stdout_line = record.stdout_queue.get_nowait()
                except queue.Empty:
                    break
                self._send_sse({"type": "console", "message": stdout_line.rstrip()})

            return_code = record.process.poll()
            if return_code is not None and not sent_process_end:
                sent_process_end = True
                self._send_sse({"type": "process_exit", "returnCode": return_code})

            if sent_process_end and record.stdout_queue.empty():
                # One more short pass catches workflow_end written immediately
                # before process termination.
                time.sleep(0.4)
                lines, offset = _read_new_log_lines(record.log_path, offset)
                for line in lines:
                    payload = _parse_json_line(line)
                    if payload:
                        self._send_sse({"type": "workflow_event", "payload": payload})
                snapshot = _build_run_snapshot(record)
                if snapshot and snapshot["snapshotKey"] != record.last_snapshot_key:
                    record.last_snapshot_key = str(snapshot["snapshotKey"])
                    self._send_sse({"type": "run_snapshot", "payload": snapshot})
                self._send_sse({"type": "stream_closed"})
                return

            if time.time() - last_heartbeat > 15:
                self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                last_heartbeat = time.time()
            time.sleep(0.5)

    def _handle_report_file(self, query: str) -> None:
        params = parse_qs(query)
        raw_path = params.get("path", [""])[0]
        render_mode = str(params.get("render", [""])[0] or "").strip().lower()
        if not raw_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing path")
            return
        path = (ROOT / unquote(raw_path)).resolve()
        if ROOT not in path.parents and path != ROOT:
            self.send_error(HTTPStatus.FORBIDDEN, "Path outside repository")
            return
        path = _preferred_business_report_path(path)
        if not path.exists() and _looks_like_business_report_request(path):
            generated = _materialize_business_report_for_request(path)
            sys.stderr.write(f"[demo-web] business report fallback for {path} => {generated}\n")
            if generated is not None:
                path = generated
                render_mode = "html"
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        if render_mode == "html" and path.suffix.lower() in {".md", ".markdown"}:
            self._serve_markdown_html(path)
            return
        if path.suffix.lower() == ".html" and _looks_like_business_report_request(path):
            self._serve_business_report_html(path)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix.lower() == ".json":
            content_type = "application/json; charset=utf-8"
        elif path.suffix.lower() in {".md", ".markdown"}:
            content_type = "text/markdown; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_business_report_html(self, path: Path) -> None:
        body = path.read_text(encoding="utf-8", errors="replace")
        body = _inline_chartjs_if_missing(body)
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_markdown_html(self, path: Path) -> None:
        body = path.read_text(encoding="utf-8", errors="replace")
        rendered = markdown.markdown(
            body,
            extensions=["extra", "fenced_code", "tables", "sane_lists"],
        )
        title = html.escape(path.stem.replace("_", " ").title())
        document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #edf2f4;
        --paper: rgba(255, 255, 255, 0.92);
        --line: rgba(167, 184, 193, 0.55);
        --text: #24323d;
        --muted: #627483;
        --accent: #0f9b9c;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        padding: 28px 18px 40px;
        background:
          linear-gradient(rgba(15, 155, 156, 0.04) 1px, transparent 1px),
          linear-gradient(90deg, rgba(15, 155, 156, 0.04) 1px, transparent 1px),
          var(--bg);
        background-size: 28px 28px, 28px 28px, auto;
        color: var(--text);
        font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      main {{
        max-width: 920px;
        margin: 0 auto;
        padding: 28px 30px;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--paper);
        box-shadow: 0 18px 44px rgba(122, 138, 147, 0.12);
      }}
      h1, h2, h3, h4 {{ line-height: 1.15; color: var(--text); }}
      h1 {{ margin-top: 0; font-size: 2rem; }}
      h2 {{ margin-top: 1.8rem; padding-top: 0.8rem; border-top: 1px solid var(--line); }}
      a {{ color: var(--accent); }}
      code, pre {{
        font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
        background: rgba(15, 155, 156, 0.06);
      }}
      code {{ padding: 0.12rem 0.3rem; border-radius: 4px; }}
      pre {{
        padding: 0.9rem 1rem;
        border-radius: 10px;
        overflow: auto;
        border: 1px solid var(--line);
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background: rgba(255, 255, 255, 0.88);
      }}
      th, td {{
        padding: 0.65rem 0.75rem;
        border: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }}
      blockquote {{
        margin: 1rem 0;
        padding: 0.2rem 0 0.2rem 1rem;
        border-left: 3px solid var(--accent);
        color: var(--muted);
      }}
    </style>
  </head>
  <body>
    <main>{rendered}</main>
  </body>
</html>
"""
        data = document.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()


def _drain_stdout(record: RunRecord) -> None:
    if record.process.stdout is None:
        return
    for line in record.process.stdout:
        record.stdout_queue.put(line)


def _read_new_log_lines(path: Path, offset: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], offset
    with path.open("rb") as fp:
        fp.seek(offset)
        data = fp.read()
        next_offset = fp.tell()
    if not data:
        return [], next_offset
    return data.decode("utf-8", errors="replace").splitlines(True), next_offset


def _parse_json_line(line: str) -> dict[str, Any] | None:
    raw = line.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _looks_like_business_report_request(path: Path) -> bool:
    name = path.name.lower()
    return "business_report" in name or "business_recommendation" in name


def _existing_file(path_str: str) -> str:
    value = str(path_str or "").strip()
    if not value:
        return ""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    candidate = _preferred_business_report_path(candidate)
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return ""


def _preferred_business_report_path(path: Path) -> Path:
    candidate = path.expanduser()
    if _looks_like_business_report_request(candidate) and candidate.suffix.lower() in {".md", ".markdown"}:
        sibling_html = candidate.with_suffix(".html")
        if sibling_html.exists() and sibling_html.is_file():
            return sibling_html.resolve()
    return candidate


def _inline_chartjs_if_missing(document: str) -> str:
    if "script type=\"application/json\" id=\"data-" not in document or "new Chart(" not in document:
        return document
    try:
        from app.agents.business_translator import _v3_read_vendored_chartjs

        chartjs_inline = _v3_read_vendored_chartjs().strip()
    except Exception:
        chartjs_inline = ""
    if not chartjs_inline:
        return document
    if chartjs_inline in document:
        return document
    bootstrap_marker = "<script>(function() {"
    injected_script = "<script>" + chartjs_inline + "</script>\n"
    if bootstrap_marker in document:
        return document.replace(bootstrap_marker, injected_script + bootstrap_marker, 1)
    head_end = "</head>"
    if head_end in document:
        return document.replace(head_end, injected_script + head_end, 1)
    return injected_script + document


def _materialize_business_report_for_request(requested_path: Path) -> Path | None:
    report_dir = requested_path.parent
    technical_report = report_dir / "technical_report.md"
    manifest_path = report_dir / "run_manifest.json"
    if not technical_report.exists() or not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    existing = _existing_file(
        str((manifest.get("report_state") or {}).get("business_report_path") or "")
    ) or _existing_file(str((manifest.get("artifacts") or {}).get("business_report_html") or ""))
    if existing:
        return _backfill_run_business_report(report_dir, Path(existing))

    inferred = _infer_global_business_report_from_manifest(manifest, technical_report)
    if inferred is not None:
        return _backfill_run_business_report(report_dir, inferred)

    generic = _latest_global_business_report()
    if generic is not None:
        return _backfill_run_business_report(report_dir, generic)

    try:
        from main import configure_runtime
        from app.agents.business_translator import run_business_translation
        from app.schemas.state_schema import GraphState

        configure_runtime(".env")
        state = GraphState()
        runtime_state = manifest.get("runtime_state") or {}
        task = manifest.get("task") or {}
        state.runtime_state.workspace_root = "."
        state.runtime_state.artifact_root = str(runtime_state.get("artifact_root") or report_dir.parent)
        state.report_state.technical_report_path = str(technical_report)
        state.task_config.dataset_root = str(task.get("dataset_root") or "")
        result = run_business_translation(state)
        report_state = result.get("report_state") or {}
        candidate = _existing_file(str(report_state.get("business_report_path") or ""))
        return _backfill_run_business_report(report_dir, Path(candidate)) if candidate else None
    except Exception:
        return None


def _backfill_run_business_report(report_dir: Path, source_path: Path) -> Path | None:
    try:
        source = _preferred_business_report_path(source_path.expanduser()).resolve()
        if not source.exists() or not source.is_file():
            return None
        report_dir.mkdir(parents=True, exist_ok=True)
        run_html_path = report_dir / "business_report.html"
        if source != run_html_path:
            shutil.copyfile(source, run_html_path)
        legacy_md_path = report_dir / "business_report.md"
        legacy_md_path.write_text(
            "\n".join(
                [
                    "# Business Report",
                    "",
                    f"Primary HTML report: [{run_html_path.name}]({run_html_path.name})",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return run_html_path.resolve()
    except Exception:
        return source_path if source_path.exists() else None


def _latest_global_business_report() -> Path | None:
    candidates = sorted(
        (ROOT / "artifacts" / "reports").glob("*business_recommendation.html"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    return None


def _infer_global_business_report_from_manifest(manifest: dict[str, Any], technical_report: Path) -> Path | None:
    try:
        from app.agents.business_translator import _v3_title_to_filename

        dataset_root = str((manifest.get("task") or {}).get("dataset_root") or "").strip()
        dataset_name = Path(dataset_root).name or Path(dataset_root).stem
        if not dataset_name:
            text = technical_report.read_text(encoding="utf-8", errors="replace")
            dataset_name = Path(text).stem or "business_report"
        candidate = ROOT / "artifacts" / "reports" / _v3_title_to_filename(f"{dataset_name} business recommendation")
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    except Exception:
        return None
    return None


def _candidate_metric_keys(metric_name: str, task_family: str = "", target_mode: str = "") -> list[str]:
    keys = [
        metric_name,
        f"cv_{metric_name}_mean",
        f"val_{metric_name}",
        f"{metric_name}_mean",
    ]
    aliases = {
        "roc_auc_macro_ovr": [
            "cv_per_target_auc_mean",
            "per_target_auc_mean",
            "cv_roc_auc_macro_ovr_mean",
            "roc_auc_macro_ovr_mean",
        ],
        "f1_macro": [
            "cv_macro_f1_mean",
            "macro_f1_mean",
        ],
        "neg_root_mean_squared_error": [
            "cv_rmse_mean",
            "rmse_mean",
        ],
    }
    keys.extend(aliases.get(metric_name, []))
    if task_family == "multilabel_classification" or target_mode == "multi_label":
        keys.extend(["cv_per_target_auc_mean", "per_target_auc_mean"])
    deduped: list[str] = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def _coerce_metric_from_mapping(metrics: dict[str, Any], candidate_keys: list[str]) -> float | None:
    for key in candidate_keys:
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                return None
    lowered = {str(key).lower(): value for key, value in metrics.items()}
    for key in candidate_keys:
        if key.lower() in lowered:
            try:
                return float(lowered[key.lower()])
            except Exception:
                return None
    return None


def _extract_primary_metric(manifest: dict[str, Any]) -> tuple[str, float | None]:
    target_spec = manifest.get("target_spec") or {}
    evaluation = target_spec.get("evaluation") or {}
    model_state = manifest.get("model_state") or {}
    metric_name = str(evaluation.get("primary_metric") or "").strip()
    validation_metrics = model_state.get("validation_metrics") or {}
    best_model = str(model_state.get("best_model_name") or "").strip()
    task_family = str(target_spec.get("task_family") or "").strip()
    target_mode = str(target_spec.get("target_mode") or "").strip()

    if not metric_name:
        return "", None

    candidate_keys = _candidate_metric_keys(metric_name, task_family, target_mode)
    if best_model and isinstance(validation_metrics.get(best_model), dict):
        best_metrics = validation_metrics.get(best_model) or {}
        best_value = _coerce_metric_from_mapping(best_metrics, candidate_keys)
        if best_value is not None:
            return metric_name, best_value

    top_level_value = _coerce_metric_from_mapping(validation_metrics, candidate_keys)
    if top_level_value is not None:
        return metric_name, top_level_value
    return metric_name, None


def _find_artifact_root(record: RunRecord) -> Path | None:
    if record.artifact_root and record.artifact_root.exists():
        return record.artifact_root

    runs_root = ROOT / "artifacts" / "runs"
    if not runs_root.exists():
        return None

    candidates = []
    prefix = f"{record.dataset_slug}_"
    for path in runs_root.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        if path.stat().st_mtime < record.started_ts - 120:
            continue
        candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    record.artifact_root = candidates[0]
    return record.artifact_root


def _build_run_snapshot(record: RunRecord) -> dict[str, Any] | None:
    artifact_root = _find_artifact_root(record)
    if artifact_root is None:
        return None

    manifest_path = artifact_root / "reports" / "run_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    metric_name, metric_value = _extract_primary_metric(manifest)
    report_state = manifest.get("report_state") or {}
    artifacts = manifest.get("artifacts") or {}
    runtime_state = manifest.get("runtime_state") or {}
    data_state = manifest.get("data_state") or {}

    report_dir = artifact_root / "reports"
    business_report = _existing_file(
        str(report_state.get("business_report_path") or "")
    ) or _existing_file(str(artifacts.get("business_report_html") or "")) or _existing_file(str(artifacts.get("business_report") or ""))
    technical_report = _existing_file(
        str(report_state.get("technical_report_path") or "")
    ) or _existing_file(str(artifacts.get("technical_report") or ""))
    if business_report:
        backfilled = _backfill_run_business_report(report_dir, Path(business_report))
        if backfilled is not None:
            business_report = str(backfilled)
    if not business_report and technical_report:
        inferred = _infer_global_business_report_from_manifest(manifest, Path(technical_report))
        if inferred is not None:
            backfilled = _backfill_run_business_report(report_dir, inferred)
            business_report = str(backfilled or inferred)

    snapshot_key = "|".join(
        [
            str(manifest_path.stat().st_mtime_ns),
            metric_name,
            "" if metric_value is None else f"{metric_value:.8f}",
            business_report,
            technical_report,
        ]
    )
    return {
        "snapshotKey": snapshot_key,
        "artifactRoot": str(artifact_root),
        "runManifestPath": str(manifest_path),
        "taskFamily": str((manifest.get("target_spec") or {}).get("task_family") or ""),
        "trainRows": ((data_state.get("train_shape") or [None]) or [None])[0],
        "totalRuntimeSec": runtime_state.get("total_runtime_sec"),
        "primaryMetric": {
            "name": metric_name,
            "value": metric_value,
            "formatted": "" if metric_value is None else f"{metric_value:.4f}",
        },
        "reportPaths": {
            "business": business_report,
            "technical": technical_report,
            "manifest": str(manifest_path),
        },
    }


def _latest_snapshot_for_task(task_id: str) -> dict[str, Any] | None:
    task = TASKS.get(task_id)
    if not task:
        return None
    dataset_slug = Path(task["dataset"]).name
    candidates = sorted(
        (ROOT / "artifacts" / "runs").glob(f"{dataset_slug}_*/reports/run_manifest.json"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        return None

    class _Record:
        def __init__(self, artifact_root: Path) -> None:
            self.artifact_root = artifact_root

    for manifest_path in candidates:
        snapshot = _build_run_snapshot(_Record(manifest_path.parent.parent))
        if snapshot:
            return snapshot
    return None


def _workflow_python() -> Path:
    """Use the project virtualenv for workflow runs when available."""
    candidates = [
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "bin" / "python3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def main() -> int:
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Serving real workflow demo at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down demo server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
