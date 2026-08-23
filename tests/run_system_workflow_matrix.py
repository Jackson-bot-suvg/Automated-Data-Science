#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _discover_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "run_multiagent.py").exists():
            return parent
    raise RuntimeError("Could not locate project root from run_system_workflow_matrix.py")


PROJECT_ROOT = _discover_project_root()
REPORT_DIR = PROJECT_ROOT / "tests" / "system_reports"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.validation_matrix import ALL_VALIDATION_DATASETS  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    task: str
    dataset: str
    target: str
    description: str
    task_family: str
    target_mode: str
    primary_acceptance: bool


PRESETS = {
    spec.key: DatasetSpec(
        key=spec.key,
        task=spec.task,
        dataset=spec.dataset,
        target=spec.target,
        description=spec.description,
        task_family=spec.task_family,
        target_mode=spec.target_mode,
        primary_acceptance=spec.primary_acceptance,
    )
    for spec in ALL_VALIDATION_DATASETS
}

DEFAULT_ENV_OVERRIDES = {
    "AGENT_READER_MAX_TOKENS": "2048",
    "AGENT_PLANNER_MAX_TOKENS": "2048",
    "AGENT_BG_REVIEWER_MAX_TOKENS": "2048",
    "AGENT_REVIEWER_MAX_TOKENS": "1024",
    "AGENT_SUMMARIZER_MAX_TOKENS": "2048",
    "AGENT_DATA_CLEANING_MAX_TOKENS": "3072",
    "AGENT_FEATURE_ENGINEERING_MAX_TOKENS": "4096",
    "AGENT_MODEL_BUILDING_MAX_TOKENS": "2048",
    "AGENT_BUSINESS_TRANSLATOR_MAX_TOKENS": "2048",
}

RUNNING_RE = re.compile(r"^\[RUNNING\] LangGraph node: ([^ ]+)")
DONE_RE = re.compile(r"^\[DONE\] ([^ ]+) completed(?:\. \[(.*)\])?$")
WORKFLOW_FAILED_RE = re.compile(r"^Workflow failed: (.*)$")
TIMING_REPORT_RE = re.compile(r"^Timing report: (.*)$")
BUSINESS_REPORT_RE = re.compile(r"^Business report: (.*)$")
REVIEW_RE = re.compile(r"review=(approved|retry|fatal)\(([^)]+)\)")
PHASE_ERROR_RE = re.compile(
    r"^\[(?P<phase>[a-z_]+)\]\s+"
    r"(?P<kind>runtime_error|tool_execution_failed|artifact_missing|missing_context|target_column_lost):\s*"
    r"(?P<message>.*)$"
)

PHASE_REVIEW_ORDER = [
    "preliminary_eda",
    "data_cleaning",
    "indepth_eda",
    "feature_engineering",
    "model_building",
]
NODE_ORDER = [
    "reader",
    "background_reviewer",
    "planner",
    "preliminary_eda",
    "reviewer",
    "data_cleaning",
    "indepth_eda",
    "feature_engineering",
    "model_building",
    "summarizer",
    "business_translator",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行所有验证数据集，并边跑边写系统稳定性 Markdown 报告。"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(PRESETS),
        help="可选：指定只跑这些数据集；默认跑 validation matrix 里的全部数据集。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=2400,
        help="单个任务的超时时间（秒）。",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def build_report_path() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return REPORT_DIR / f"system-matrix-{stamp}.md"


def write_report_header(report_path: Path, datasets: list[DatasetSpec], timeout: int) -> None:
    lines = [
        "# 系统稳定性批量测试报告",
        "",
        f"- 开始时间：{utc_now()}",
        f"- Commit：`{git_commit()}`",
        f"- 单任务超时：`{timeout}s`",
        f"- 数据集数量：`{len(datasets)}`",
        "- 说明：每个任务结束后立即追加一节记录，不等待全量跑完。",
        "",
        "## 本次任务列表",
        "",
    ]
    for spec in datasets:
        tier = "primary" if spec.primary_acceptance else "smoke"
        lines.append(
            f"- `{spec.key}`: {spec.description} [{tier}] target=`{spec.target}`"
        )
    lines.extend(["", "---", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def append_report(report_path: Path, text: str) -> None:
    with report_path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def run_workflow(dataset: str, timeout: int) -> tuple[int, str]:
    env = os.environ.copy()
    for key, value in DEFAULT_ENV_OVERRIDES.items():
        env.setdefault(key, value)
    cmd = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-u", "run_multiagent.py", dataset]
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None

    deadline = time.monotonic() + timeout
    lines: list[str] = []
    timed_out = False

    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
            if time.monotonic() > deadline:
                timed_out = True
                process.kill()
                break
    finally:
        remaining = process.stdout.read()
        if remaining:
            sys.stdout.write(remaining)
            sys.stdout.flush()
            lines.append(remaining)
        process.stdout.close()

    returncode = process.wait()
    if timed_out:
        return 124, "".join(lines)
    return returncode, "".join(lines)


def list_run_dirs() -> set[Path]:
    root = PROJECT_ROOT / "artifacts" / "runs"
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def discover_run_dir(output: str, before: set[Path]) -> Path | None:
    timing_match = TIMING_REPORT_RE.search(output)
    if timing_match:
        timing_path = Path(timing_match.group(1).strip()).expanduser()
        if not timing_path.is_absolute():
            timing_path = (PROJECT_ROOT / timing_path).resolve()
        if timing_path.exists():
            return timing_path.parent.parent.resolve()

    business_match = BUSINESS_REPORT_RE.search(output)
    if business_match:
        report_path = Path(business_match.group(1).strip()).expanduser()
        if not report_path.is_absolute():
            report_path = (PROJECT_ROOT / report_path).resolve()
        if report_path.exists():
            return report_path.parent.parent.resolve()

    after = list_run_dirs()
    candidates = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_output_events(output: str) -> dict[str, Any]:
    node_counts: dict[str, int] = {}
    last_running = ""
    failure_messages: list[str] = []
    signal_lines: list[str] = []
    phase_error_events: dict[str, list[str]] = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()
        phase_error = PHASE_ERROR_RE.match(line)
        if phase_error:
            phase = phase_error.group("phase")
            kind = phase_error.group("kind")
            message = phase_error.group("message").strip()
            phase_error_events.setdefault(phase, []).append(f"{kind}: {message}")
            signal_lines.append(line)
            continue

        running = RUNNING_RE.match(line)
        if running:
            last_running = running.group(1)
            signal_lines.append(line)
            continue

        done = DONE_RE.match(line)
        if done:
            node = done.group(1)
            node_counts[node] = node_counts.get(node, 0) + 1
            signal_lines.append(line)
            continue

        failed = WORKFLOW_FAILED_RE.match(line)
        if failed:
            failure_messages.append(failed.group(1).strip())
            signal_lines.append(line)
            continue

        if (
            "timed out/cancelled" in line
            or line.startswith("Timing report:")
            or line.startswith("Business report:")
            or line.startswith("REPORT::")
        ):
            signal_lines.append(line)

    return {
        "node_counts": node_counts,
        "last_running": last_running,
        "failure_messages": failure_messages,
        "phase_error_events": phase_error_events,
        "tail_lines": signal_lines[-15:],
    }


def extract_primary_metric(manifest: dict[str, Any]) -> tuple[str, str]:
    target_spec = manifest.get("target_spec") or {}
    evaluation = target_spec.get("evaluation") or {}
    model_state = manifest.get("model_state") or {}
    primary = str(evaluation.get("primary_metric") or "").strip()
    metrics = dict(model_state.get("validation_metrics") or {})
    if not primary:
        return "", ""

    candidates = [
        primary,
        f"cv_{primary}_mean",
        f"val_{primary}",
        f"{primary}_mean",
    ]
    best_model_name = str(model_state.get("best_model_name") or "").strip()
    if best_model_name and isinstance(metrics.get(best_model_name), dict):
        best_metrics = dict(metrics.get(best_model_name) or {})
        for key in candidates:
            if key in best_metrics:
                try:
                    return primary, f"{float(best_metrics[key]):.4f}"
                except Exception:
                    return primary, str(best_metrics[key])

    for key in candidates:
        if key in metrics:
            try:
                return primary, f"{float(metrics[key]):.4f}"
            except Exception:
                return primary, str(metrics[key])
    return primary, ""


def _phase_blocking_notes(
    *,
    phase: str,
    manifest: dict[str, Any] | None,
    output_meta: dict[str, Any],
    latest_blocking: str,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    issues: list[str] = []

    if manifest:
        review_state = manifest.get("review_state") or {}
        review_history_summary = list(review_state.get("review_history_summary") or [])
        for record in review_history_summary:
            if not isinstance(record, dict):
                continue
            if str(record.get("phase") or "").strip() != phase:
                continue
            attempt = int(record.get("attempt") or 0)
            status = str(record.get("status") or "").strip()
            blocking_failures = list(record.get("blocking_failures") or [])
            warnings = list(record.get("warnings") or [])
            fragments: list[str] = []
            if blocking_failures:
                fragments.append("; ".join(str(item) for item in blocking_failures[:3]))
            if warnings:
                fragments.append("warnings: " + "; ".join(str(item) for item in warnings[:2]))
            if fragments:
                notes.append(f"attempt {attempt} ({status}): " + " ; ".join(fragments))
                issues.extend(f"{phase}: {text}" for text in blocking_failures[:1])

        fatal_errors = list(review_state.get("fatal_errors") or [])
        phase_fatal_errors = []
        prefix = f"[{phase}] "
        for item in fatal_errors:
            text = str(item)
            if text.startswith(prefix):
                phase_fatal_errors.append(text[len(prefix):])
        for idx, item in enumerate(phase_fatal_errors, start=1):
            notes.append(f"phase_error {idx}: {item}")
            issues.append(f"{phase}: {item}")

    phase_error_events = dict(output_meta.get("phase_error_events") or {})
    for idx, item in enumerate(phase_error_events.get(phase, []), start=1):
        rendered = f"phase_error_log {idx}: {item}"
        if rendered not in notes:
            notes.append(rendered)
            issues.append(f"{phase}: {item}")

    if latest_blocking and not any(latest_blocking in note for note in notes):
        notes.append(f"latest: {latest_blocking}")
        issues.append(f"{phase}: {latest_blocking}")

    return "<br>".join(notes), issues


def summarize_phase_reviews(
    run_dir: Path,
    manifest: dict[str, Any] | None,
    output_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    review_dir = run_dir / "phase_reviews"
    for phase in PHASE_REVIEW_ORDER:
        path = review_dir / f"{phase}.json"
        if not path.exists():
            rows.append(
                {
                    "phase": phase,
                    "status": "not_reached",
                    "attempt": 0,
                    "retries": 0,
                    "blocking": "",
                }
            )
            continue
        payload = load_json(path)
        attempt = int(payload.get("attempt") or 0)
        blocking_failures = list(payload.get("blocking_failures") or [])
        blocking_note, phase_issues = _phase_blocking_notes(
            phase=phase,
            manifest=manifest,
            output_meta=output_meta,
            latest_blocking=blocking_failures[0] if blocking_failures else "",
        )
        rows.append(
            {
                "phase": phase,
                "status": str(payload.get("status") or "unknown"),
                "attempt": attempt,
                "retries": max(attempt - 1, 0),
                "blocking": blocking_note,
            }
        )
        issues.extend(phase_issues[:3])
    return rows, issues


def summarize_node_timings(
    output_meta: dict[str, Any],
    manifest: dict[str, Any] | None,
    timing_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    phase_runtime = {}
    if timing_payload:
        phase_runtime = dict(timing_payload.get("phase_durations_sec") or {})
    elif manifest:
        runtime_state = manifest.get("runtime_state") or {}
        phase_runtime = dict(runtime_state.get("phase_runtime_sec") or {})

    node_counts = dict(output_meta.get("node_counts") or {})
    rows: list[dict[str, Any]] = []
    for node in NODE_ORDER:
        status = "completed" if node_counts.get(node, 0) > 0 else "not_reached"
        duration = ""
        if node in phase_runtime:
            try:
                duration = f"{float(phase_runtime[node]):.4f}"
            except Exception:
                duration = str(phase_runtime[node])
        rows.append(
            {
                "node": node,
                "status": status,
                "count": node_counts.get(node, 0),
                "duration": duration,
            }
        )
    return rows


def task_outcome(
    *,
    returncode: int,
    output_meta: dict[str, Any],
    manifest: dict[str, Any] | None,
    technical_report_exists: bool,
    business_report_exists: bool,
) -> tuple[str, str, str]:
    if returncode == 0 and technical_report_exists:
        if business_report_exists:
            return "通过", "full_workflow_completed", ""
        return "部分通过", "technical_only_completed", "business_report 未生成"

    latest_review_status = ""
    current_phase = ""
    fatal_error = ""
    if manifest:
        review_state = manifest.get("review_state") or {}
        latest_review_status = str(review_state.get("latest_review_status") or "")
        current_phase = str(manifest.get("current_phase") or "")
        fatal_errors = list(review_state.get("fatal_errors") or [])
        fatal_error = fatal_errors[0] if fatal_errors else ""

    stop_reason = fatal_error
    if not stop_reason:
        failure_messages = list(output_meta.get("failure_messages") or [])
        if failure_messages:
            stop_reason = failure_messages[0]
    if not stop_reason and latest_review_status:
        stop_reason = latest_review_status

    stop_module = current_phase or str(output_meta.get("last_running") or "unknown")
    return "失败", stop_module, stop_reason


def format_phase_review_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Phase | Status | Attempt | Retries | Blocking note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        blocking = str(row["blocking"] or "").replace("\n", " ").strip()
        lines.append(
            f"| `{row['phase']}` | `{row['status']}` | {row['attempt']} | {row['retries']} | {blocking or '-'} |"
        )
    return lines


def format_node_timing_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Node | Status | Count | Time (s) |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['node']}` | `{row['status']}` | {row['count']} | {row['duration'] or '-'} |"
        )
    return lines


def build_task_section(
    *,
    spec: DatasetSpec,
    report_index: int,
    returncode: int,
    output: str,
    run_dir: Path | None,
) -> str:
    output_meta = parse_output_events(output)
    manifest = None
    timing_payload = None
    technical_report_exists = False
    business_report_exists = False
    primary_metric_name = ""
    primary_metric_value = ""
    run_id = ""
    current_phase = ""
    artifact_root = ""
    fatal_errors: list[str] = []
    phase_rows: list[dict[str, Any]] = []
    phase_issues: list[str] = []

    if run_dir:
        manifest_path = run_dir / "reports" / "run_manifest.json"
        timing_path = run_dir / "reports" / "run_timing.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            run_id = str(manifest.get("run_id") or "")
            current_phase = str(manifest.get("current_phase") or "")
            runtime_state = manifest.get("runtime_state") or {}
            artifact_root = str(runtime_state.get("artifact_root") or run_dir)
            review_state = manifest.get("review_state") or {}
            fatal_errors = list(review_state.get("fatal_errors") or [])
            primary_metric_name, primary_metric_value = extract_primary_metric(manifest)
            phase_rows, phase_issues = summarize_phase_reviews(
                run_dir,
                manifest,
                output_meta,
            )
            report_state = manifest.get("report_state") or {}
            technical_report_path = str(report_state.get("technical_report_path") or "")
            if not technical_report_path:
                technical_report_path = str((manifest.get("artifacts") or {}).get("technical_report") or "")
            if technical_report_path:
                technical_report_exists = Path(technical_report_path).exists()

            business_report_path = str(report_state.get("business_report_path") or "")
            if not business_report_path:
                business_report_path = str((manifest.get("artifacts") or {}).get("business_report") or "")
            if business_report_path:
                business_report_exists = Path(business_report_path).exists()
        if timing_path.exists():
            timing_payload = load_json(timing_path)

    if not business_report_exists:
        business_match = BUSINESS_REPORT_RE.search(output)
        if business_match:
            candidate = Path(business_match.group(1).strip()).expanduser()
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            business_report_exists = candidate.exists()

    node_rows = summarize_node_timings(output_meta, manifest, timing_payload)
    result_label, stop_module, stop_reason = task_outcome(
        returncode=returncode,
        output_meta=output_meta,
        manifest=manifest,
        technical_report_exists=technical_report_exists,
        business_report_exists=business_report_exists,
    )
    total_runtime = ""
    if timing_payload:
        try:
            total_runtime = f"{float(timing_payload.get('total_runtime_sec') or 0.0):.4f}"
        except Exception:
            total_runtime = str(timing_payload.get("total_runtime_sec") or "")

    notes: list[str] = []
    if stop_reason:
        notes.append(stop_reason)
    notes.extend(fatal_errors[:2])
    notes.extend(phase_issues[:3])
    if not notes and returncode == 124:
        notes.append("任务超时被脚本中断。")
    if not notes and returncode != 0:
        notes.append("任务以非零状态退出，但没有提炼出更具体的错误摘要。")

    lines = [
        f"## {report_index}. `{spec.key}`",
        "",
        f"- 记录时间：{utc_now()}",
        f"- 结果：**{result_label}**",
        f"- 数据集类型：`{spec.description}`",
        f"- 任务族 / 目标模式：`{spec.task_family}` / `{spec.target_mode}`",
        f"- run_id：`{run_id or 'unknown'}`",
        f"- 当前或停止阶段：`{current_phase or stop_module or 'unknown'}`",
        f"- 返回码：`{returncode}`",
        f"- Artifact root：`{artifact_root or (str(run_dir) if run_dir else 'unknown')}`",
        f"- 总耗时：`{total_runtime or '-'}` s",
    ]
    if primary_metric_name:
        metric_text = primary_metric_value or "not available"
        lines.append(f"- 主指标：`{primary_metric_name}` = `{metric_text}`")
    else:
        lines.append("- 主指标：`not available`")
    lines.extend(
        [
            f"- Technical report：`{'yes' if technical_report_exists else 'no'}`",
            f"- Business report：`{'yes' if business_report_exists else 'no'}`",
            "",
            "### Phase Gates",
            *format_phase_review_table(
                phase_rows
                or [
                    {
                        "phase": phase,
                        "status": "unknown",
                        "attempt": 0,
                        "retries": 0,
                        "blocking": "",
                    }
                    for phase in PHASE_REVIEW_ORDER
                ]
            ),
            "",
            "### Node Timings",
            *format_node_timing_table(node_rows),
            "",
            "### 备注",
        ]
    )
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- 无显式阻塞问题。")

    tail_lines = list(output_meta.get("tail_lines") or [])
    if tail_lines:
        lines.extend(
            [
                "",
                "### 日志尾部",
                "```text",
                *tail_lines,
                "```",
            ]
        )
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def resolve_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.datasets:
        return [PRESETS[key] for key in args.datasets]
    return [PRESETS[spec.key] for spec in ALL_VALIDATION_DATASETS]


def main() -> int:
    args = parse_args()
    datasets = resolve_datasets(args)
    report_path = build_report_path()
    write_report_header(report_path, datasets, args.timeout)
    print(f"批量测试报告：{report_path}", flush=True)

    failures = 0
    for index, spec in enumerate(datasets, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[BATCH] Running {spec.key} ({index}/{len(datasets)})", flush=True)
        before = list_run_dirs()
        try:
            returncode, output = run_workflow(spec.key, args.timeout)
        except Exception as exc:
            returncode = 1
            output = f"Workflow failed: runner_exception: {exc}\n"
        run_dir = discover_run_dir(output, before)
        section = build_task_section(
            spec=spec,
            report_index=index,
            returncode=returncode,
            output=output,
            run_dir=run_dir,
        )
        append_report(report_path, section)
        if returncode != 0:
            failures += 1

    append_report(
        report_path,
        "\n".join(
            [
                "## 结束信息",
                "",
                f"- 结束时间：{utc_now()}",
                f"- 任务总数：`{len(datasets)}`",
                f"- 失败数：`{failures}`",
                "",
            ]
        ),
    )
    print(f"REPORT::{report_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
