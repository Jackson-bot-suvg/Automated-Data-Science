#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


def _discover_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "run_multiagent.py").exists():
            return parent
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _discover_project_root()
REPORT_DIR = PROJECT_ROOT / "tests" / "system_reports"
LADS_ROOT = PROJECT_ROOT / "_external" / "lightautods_eval" / "LADS"
LADS_HELPER = PROJECT_ROOT / "tests" / "helpers" / "run_lads_experiment.py"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightAutoDS across the validation matrix and merge with existing main/AIDE results.")
    parser.add_argument("datasets", nargs="*", choices=sorted(PRESETS))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--lads-timeout", type=int, default=2400)
    parser.add_argument("--baseline-json", default="")
    return parser.parse_args()


def resolve_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.datasets:
        return [PRESETS[key] for key in args.datasets]
    return [PRESETS[spec.key] for spec in ALL_VALIDATION_DATASETS]


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


def build_report_paths() -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    md = REPORT_DIR / f"framework-matrix-threeway-{stamp}.md"
    js = REPORT_DIR / f"framework-matrix-threeway-{stamp}.json"
    return md, js


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_score(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def infer_metric_direction(metric_name: str) -> str:
    metric = metric_name.lower().strip()
    if metric.startswith("neg_"):
        return "maximize"
    if any(token in metric for token in ("error", "loss", "rmse", "mae", "mse")):
        return "minimize"
    return "maximize"


def build_lads_env(env_file: str) -> dict[str, str]:
    values = dotenv_values(env_file)
    base_url = str(values.get("LLM_MINIMAX_BASE_URL") or values.get("OPENAI_BASE_URL") or "").strip()
    api_key = str(values.get("LLM_MINIMAX_API_KEY") or values.get("OPENAI_API_KEY") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError("MiniMax/OpenAI-compatible API credentials are missing from env file.")

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = base_url
    return env


def discover_baseline_json(explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found: {path}")
        return path

    candidates = sorted(REPORT_DIR.glob("framework-matrix-*-corrected.json"))
    if not candidates:
        raise FileNotFoundError("No corrected two-framework baseline JSON found in tests/system_reports.")
    return candidates[-1]


def run_lads_dataset(spec: DatasetSpec, metric_name: str, env_file: str, timeout: int) -> dict[str, Any]:
    env = build_lads_env(env_file)
    output_json = REPORT_DIR / "tmp" / f"lads-{spec.key}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(LADS_ROOT / ".venv" / "bin" / "python"),
        str(LADS_HELPER),
        "--repo-root",
        str(LADS_ROOT),
        "--data-dir",
        str((PROJECT_ROOT / spec.dataset).resolve()),
        "--task",
        spec.task,
        "--target",
        spec.target,
        "--metric-name",
        metric_name,
        "--task-family",
        spec.task_family,
        "--description",
        spec.description,
        "--output-json",
        str(output_json),
    ]
    started = time.time()
    stdout_tail = ""
    stderr_tail = ""
    returncode = 0
    payload: dict[str, Any] = {}
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        stdout_tail = (proc.stdout or "")[-4000:]
        stderr_tail = (proc.stderr or "")[-2000:]
        returncode = proc.returncode
        payload = load_json(output_json) if output_json.exists() else {}
    except subprocess.TimeoutExpired as exc:
        stdout_tail = (exc.stdout or "")[-4000:] if exc.stdout else ""
        stderr_tail = (exc.stderr or "")[-2000:] if exc.stderr else ""
        returncode = -1
        payload = {
            "status": "FAIL",
            "runtime_sec": round(time.time() - started, 4),
            "metric_name": metric_name,
            "error": f"TimeoutExpired: exceeded {timeout}s",
            "run_dir": str(output_json.with_suffix("")),
        }
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "framework": "lads",
        "status": payload.get("status", "FAIL"),
        "returncode": returncode,
        "runtime_sec": float(payload.get("runtime_sec") or round(time.time() - started, 4)),
        "metric_name": metric_name,
        "metric_value": payload.get("metric_value"),
        "run_dir": str(payload.get("run_dir") or ""),
        "prediction_file": str(payload.get("prediction_file") or ""),
        "prediction_exists": bool(payload.get("prediction_exists")),
        "route": str(payload.get("route") or ""),
        "nodes_visited": list(payload.get("nodes_visited") or []),
        "feedback_count": int(payload.get("feedback_count") or 0),
        "human_understanding_count": int(payload.get("human_understanding_count") or 0),
        "generated_code_length": int(payload.get("generated_code_length") or 0),
        "metric_trace": list(payload.get("metric_trace") or []),
        "analysis": str(payload.get("final_message") or ""),
        "error": str(payload.get("error") or ""),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def determine_best_framework(metric_name: str, scores: dict[str, float | None]) -> tuple[str, float | None]:
    available = {name: value for name, value in scores.items() if value is not None}
    if not available:
        return "not_comparable", None
    direction = infer_metric_direction(metric_name)
    ordered = sorted(
        available.items(),
        key=lambda item: item[1],
        reverse=(direction == "maximize"),
    )
    winner, winner_score = ordered[0]
    if len(ordered) == 1:
        return winner, None
    runner_up_score = ordered[1][1]
    return winner, abs(float(winner_score) - float(runner_up_score))


def build_snapshot(rows: list[dict[str, Any]]) -> list[str]:
    comparable = [row for row in rows if row.get("best_framework") != "not_comparable"]
    main_wins = sum(1 for row in comparable if row.get("best_framework") == "main")
    aide_wins = sum(1 for row in comparable if row.get("best_framework") == "aide")
    lads_wins = sum(1 for row in comparable if row.get("best_framework") == "lads")
    main_total_time = sum(float((row.get("main") or {}).get("runtime_sec") or 0.0) for row in rows)
    aide_total_time = sum(float((row.get("aide") or {}).get("runtime_sec") or 0.0) for row in rows)
    lads_total_time = sum(float((row.get("lads") or {}).get("runtime_sec") or 0.0) for row in rows)
    main_retries = sum(int((row.get("main") or {}).get("feature_engineering_retries") or 0) for row in rows)
    aide_buggy_nodes = sum(int((row.get("aide") or {}).get("buggy_nodes") or 0) for row in rows)
    lads_failures = sum(1 for row in rows if (row.get("lads") or {}).get("status") != "PASS")
    return [
        "## Snapshot",
        "",
        f"- Comparable tasks: `{len(comparable)}/{len(rows)}`",
        f"- Win count: `main={main_wins}`, `aide={aide_wins}`, `lightautods={lads_wins}`",
        f"- Total runtime: `main={fmt_score(main_total_time)}s`, `aide={fmt_score(aide_total_time)}s`, `lightautods={fmt_score(lads_total_time)}s`",
        f"- Stability signals: `main_feature_retries={main_retries}`, `aide_buggy_nodes={aide_buggy_nodes}`, `lightautods_failures={lads_failures}`",
        "",
    ]


def write_report(report_path: Path, json_path: Path, rows: list[dict[str, Any]], baseline_path: Path, timeout: int) -> None:
    lines = [
        "# Three-Framework Matrix Comparison",
        "",
        f"- Generated: {utc_now()}",
        f"- Commit: `{git_commit()}`",
        f"- Baseline: `{baseline_path}`",
        f"- LightAutoDS timeout: `{timeout}s`",
        "",
    ]
    if rows:
        lines.extend(build_snapshot(rows))
    lines.extend(
        [
            "## Summary",
            "",
            "| Dataset | Primary Metric | Main | AIDE | LightAutoDS | Best | Advantage | Main Time (s) | AIDE Time (s) | LightAutoDS Time (s) | LightAutoDS Route |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | `{row['metric_name']}` | {fmt_score(row['main'].get('metric_value'))} | "
            f"{fmt_score(row['aide'].get('metric_value'))} | {fmt_score(row['lads'].get('metric_value'))} | "
            f"`{row['best_framework']}` | {fmt_score(row['advantage'])} | "
            f"{fmt_score(row['main'].get('runtime_sec'))} | {fmt_score(row['aide'].get('runtime_sec'))} | "
            f"{fmt_score(row['lads'].get('runtime_sec'))} | `{row['lads'].get('route') or '-'}` |"
        )

    for idx, row in enumerate(rows, start=1):
        main = row["main"]
        aide = row["aide"]
        lads = row["lads"]
        lines.extend(
            [
                "",
                f"## {idx}. `{row['dataset']}`",
                "",
                f"- Description: `{row['description']}`",
                f"- Primary metric: `{row['metric_name']}`",
                f"- Best framework: `{row['best_framework']}`",
                f"- Winning margin: `{fmt_score(row['advantage'])}`",
                "",
                "### Main",
                f"- Status: `{main.get('status')}`",
                f"- Metric: `{fmt_score(main.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(main.get('runtime_sec'))}`",
                f"- Technical report: `{main.get('technical_report') or '-'}`",
                f"- Prediction file: `{main.get('prediction_file') or '-'}`",
                "",
                "### AIDE",
                f"- Status: `{aide.get('status')}`",
                f"- Metric: `{fmt_score(aide.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(aide.get('runtime_sec'))}`",
                f"- Buggy nodes: `{aide.get('buggy_nodes', 0)}` / total nodes `{aide.get('nodes_total', 0)}`",
                f"- Prediction file: `{aide.get('prediction_file') or '-'}`",
                "",
                "### LightAutoDS",
                f"- Status: `{lads.get('status')}`",
                f"- Metric: `{fmt_score(lads.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(lads.get('runtime_sec'))}`",
                f"- Route: `{lads.get('route') or '-'}`",
                f"- Feedback count: `{lads.get('feedback_count', 0)}`",
                f"- Prediction file: `{lads.get('prediction_file') or '-'}`",
            ]
        )
        if lads.get("error"):
            lines.append(f"- LightAutoDS error: `{lads['error']}`")
        if lads.get("analysis"):
            lines.append(f"- LightAutoDS final summary: {lads['analysis']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets = resolve_datasets(args)
    baseline_path = discover_baseline_json(args.baseline_json)
    baseline_rows = {row["dataset"]: row for row in load_json(baseline_path)}

    report_path, json_path = build_report_paths()
    rows: list[dict[str, Any]] = []

    for spec in datasets:
        if spec.key not in baseline_rows:
            raise KeyError(f"Dataset {spec.key} missing from baseline file {baseline_path}")
        baseline_row = baseline_rows[spec.key]
        metric_name = str(baseline_row.get("metric_name") or "").strip()
        print(f"[LADS-COMPARE] {spec.key}", flush=True)
        lads_result = run_lads_dataset(spec, metric_name, args.env_file, args.lads_timeout)
        scores = {
            "main": baseline_row["main"].get("metric_value"),
            "aide": baseline_row["aide"].get("metric_value"),
            "lads": lads_result.get("metric_value"),
        }
        best_framework, advantage = determine_best_framework(metric_name, scores)
        rows.append(
            {
                "dataset": spec.key,
                "description": spec.description,
                "metric_name": metric_name,
                "best_framework": best_framework,
                "advantage": advantage,
                "main": baseline_row["main"],
                "aide": baseline_row["aide"],
                "lads": lads_result,
            }
        )
        write_report(report_path, json_path, rows, baseline_path, args.lads_timeout)
        print(f"[LADS-COMPARE] report updated: {report_path}", flush=True)

    print(f"REPORT::{report_path}")
    print(f"JSON::{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
