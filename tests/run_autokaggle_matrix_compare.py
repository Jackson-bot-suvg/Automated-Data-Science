#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _discover_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "run_multiagent.py").exists():
            return parent
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _discover_project_root()
REPORT_DIR = PROJECT_ROOT / "tests" / "system_reports"
TMP_DIR = REPORT_DIR / "tmp"
AUTOKAGGLE_ROOT = PROJECT_ROOT / "_external" / "autokaggle_eval" / "AutoKaggle"
AUTOKAGGLE_PYTHON = PROJECT_ROOT / "_external" / "autokaggle_eval" / ".venv" / "bin" / "python"
AUTOKAGGLE_HELPER = PROJECT_ROOT / "tests" / "helpers" / "run_autokaggle_experiment.py"
PRIMARY_MD = REPORT_DIR / "framework-matrix-20260505-051038-corrected-with-lightautods.md"
ENGLISH_MD = REPORT_DIR / "framework-matrix-20260505-051038-corrected-with-lightautods-en.md"
PRIMARY_JSON = REPORT_DIR / "framework-matrix-20260505-051038-corrected-with-lightautods.json"

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


DATASETS = {
    spec.key: DatasetSpec(
        key=spec.key,
        task=spec.task,
        dataset=spec.dataset,
        target=spec.target,
        description=spec.description,
        task_family=spec.task_family,
    )
    for spec in ALL_VALIDATION_DATASETS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AutoKaggle on the validation matrix and merge results into the existing three-framework report."
    )
    parser.add_argument("datasets", nargs="*", choices=sorted(DATASETS))
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--timeout-sec", type=int, default=2400)
    parser.add_argument("--valid-size", type=float, default=0.2)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def determine_best_framework(metric_name: str, row: dict[str, Any]) -> tuple[str, float | None]:
    scores: dict[str, float] = {}
    for framework in ("main", "aide", "lightautods_lama", "autokaggle"):
        payload = row.get(framework) or {}
        value = payload.get("metric_value")
        if value is None:
            continue
        scores[framework] = float(value)
    if not scores:
        return "not_comparable", None
    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=(infer_metric_direction(metric_name) == "maximize"),
    )
    winner, winner_score = ordered[0]
    if len(ordered) == 1:
        return winner, None
    return winner, abs(float(winner_score) - float(ordered[1][1]))


def run_autokaggle(spec: DatasetSpec, env_file: str, timeout_sec: int, valid_size: float) -> dict[str, Any]:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_json = TMP_DIR / f"autokaggle-{spec.key}-{timestamp}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    metric_name = next(dataset.metric_hint for dataset in ALL_VALIDATION_DATASETS if dataset.key == spec.key)
    cmd = [
        str(AUTOKAGGLE_PYTHON),
        str(AUTOKAGGLE_HELPER),
        "--repo-root",
        str(PROJECT_ROOT),
        "--autokaggle-root",
        str(AUTOKAGGLE_ROOT),
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
        "--env-file",
        env_file,
        "--timeout-sec",
        str(timeout_sec),
        "--valid-size",
        str(valid_size),
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    payload = load_json(output_json)
    return {
        "framework": "autokaggle",
        "status": payload.get("status", "FAIL"),
        "returncode": proc.returncode,
        "runtime_sec": payload.get("runtime_sec"),
        "metric_name": payload.get("metric_name", metric_name),
        "metric_value": payload.get("metric_value"),
        "run_dir": payload.get("competition_dir", ""),
        "prediction_file": payload.get("prediction_file", ""),
        "prediction_exists": payload.get("prediction_exists", False),
        "valid_output": payload.get("valid_output", False),
        "valid_output_issues": payload.get("valid_output_issues", []),
        "validation_rows": payload.get("validation_rows"),
        "route": payload.get("route", "multi_agent_validation_harness"),
        "log_file": payload.get("log_file", ""),
        "research_report": payload.get("research_report", ""),
        "model_name": payload.get("model_name", ""),
        "stdout_tail": payload.get("stdout_tail", ""),
        "stderr_tail": payload.get("stderr_tail", ""),
        "error": payload.get("error", ""),
    }


def render_snapshot(rows: list[dict[str, Any]], generated_at: str) -> list[str]:
    comparable = [row for row in rows if row.get("better") != "not_comparable"]
    winners = {name: 0 for name in ("main", "aide", "lightautods_lama", "autokaggle")}
    for row in comparable:
        winners[row["better"]] = winners.get(row["better"], 0) + 1
    runtimes = {
        name: sum(float((row.get(name) or {}).get("runtime_sec") or 0.0) for row in rows)
        for name in ("main", "aide", "lightautods_lama", "autokaggle")
    }
    return [
        f"- Generated: {generated_at}",
        f"- Refreshed from baseline: `{PRIMARY_JSON}`",
        "- Main, AIDE, and LightAutoDS rows are reused from the prior report; only AutoKaggle was newly executed.",
        "",
        "## Snapshot",
        "",
        f"- Comparable tasks: `{len(comparable)}/{len(rows)}`",
        (
            f"- Win count: `main={winners['main']}`, `aide={winners['aide']}`, "
            f"`lightautods_lama={winners['lightautods_lama']}`, `autokaggle={winners['autokaggle']}`"
        ),
        (
            f"- Total runtime: `main={runtimes['main']:.4f}s`, `aide={runtimes['aide']:.4f}s`, "
            f"`lightautods_lama={runtimes['lightautods_lama']:.4f}s`, `autokaggle={runtimes['autokaggle']:.4f}s`"
        ),
        "- AutoKaggle route: original multi-agent workflow with a held-out validation split exposed as the temporary competition test set, then externally rescored on the main-project benchmark metric.",
        "- AutoKaggle provider: `MiniMax 2.7` through the repo's OpenAI-compatible API path, with compatibility patches limited to the isolated `_external/autokaggle_eval` copy.",
        "",
    ]


def render_markdown(title: str, rows: list[dict[str, Any]], generated_at: str, english: bool) -> str:
    lines = [title, ""]
    if english:
        lines.append("- Language: `English`")
    lines.extend(render_snapshot(rows, generated_at))
    lines.extend(
        [
            "## Summary",
            "",
            "| Dataset | Primary Metric | Main | AIDE | LightAutoDS | AutoKaggle | Better | Advantage | Main Time (s) | AIDE Time (s) | LightAutoDS Time (s) | AutoKaggle Time (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| `{dataset}` | `{metric}` | {main} | {aide} | {lads} | {autokaggle} | `{better}` | {advantage} | {main_t} | {aide_t} | {lads_t} | {autokaggle_t} |".format(
                dataset=row["dataset"],
                metric=row["metric_name"],
                main=fmt_score((row.get("main") or {}).get("metric_value")),
                aide=fmt_score((row.get("aide") or {}).get("metric_value")),
                lads=fmt_score((row.get("lightautods_lama") or {}).get("metric_value")),
                autokaggle=fmt_score((row.get("autokaggle") or {}).get("metric_value")),
                better=row.get("better", "not_comparable"),
                advantage=fmt_score(row.get("advantage")),
                main_t=fmt_score((row.get("main") or {}).get("runtime_sec")),
                aide_t=fmt_score((row.get("aide") or {}).get("runtime_sec")),
                lads_t=fmt_score((row.get("lightautods_lama") or {}).get("runtime_sec")),
                autokaggle_t=fmt_score((row.get("autokaggle") or {}).get("runtime_sec")),
            )
        )

    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                "",
                f"## {index}. `{row['dataset']}`",
                "",
                f"- Description: `{row['description']}`",
                f"- Primary metric: `{row['metric_name']}`",
                f"- Better framework: `{row.get('better', 'not_comparable')}`",
                f"- Metric advantage: `{fmt_score(row.get('advantage'))}`",
                "",
                "### Main",
            ]
        )
        main = row.get("main") or {}
        lines.extend(
            [
                f"- Status: `{main.get('status', 'FAIL')}`",
                f"- Metric: `{fmt_score(main.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(main.get('runtime_sec'))}`",
                f"- Technical report: `{main.get('technical_report', '-')}`",
                f"- Prediction file: `{main.get('prediction_file', '-')}`",
            ]
        )
        if main.get("stdout_tail"):
            lines.append(f"- Notes: `{str(main['stdout_tail'])[:240]}`")

        lines.extend(["", "### AIDE"])
        aide = row.get("aide") or {}
        lines.extend(
            [
                f"- Status: `{aide.get('status', 'FAIL')}`",
                f"- Metric: `{fmt_score(aide.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(aide.get('runtime_sec'))}`",
                f"- Prediction file: `{aide.get('prediction_file', '-')}`",
            ]
        )

        lines.extend(["", "### LightAutoDS"])
        lads = row.get("lightautods_lama") or {}
        lines.extend(
            [
                f"- Status: `{lads.get('status', 'FAIL')}`",
                f"- Metric: `{fmt_score(lads.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(lads.get('runtime_sec'))}`",
                f"- Route: `{lads.get('route', '-')}`",
                f"- Internal config: `{json.dumps(lads.get('internal_config', {}), ensure_ascii=False)}`",
                f"- Validation predictions: `{lads.get('validation_predictions_file', '-')}`",
                f"- Prediction file: `{lads.get('prediction_file', '-')}`",
            ]
        )

        lines.extend(["", "### AutoKaggle"])
        autokaggle = row.get("autokaggle") or {}
        lines.extend(
            [
                f"- Status: `{autokaggle.get('status', 'FAIL')}`",
                f"- Metric: `{fmt_score(autokaggle.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(autokaggle.get('runtime_sec'))}`",
                f"- Route: `{autokaggle.get('route', '-')}`",
                f"- Model: `{autokaggle.get('model_name', '-')}`",
                f"- Valid output: `{autokaggle.get('valid_output', False)}`",
                f"- Prediction file: `{autokaggle.get('prediction_file', '-')}`",
                f"- Log file: `{autokaggle.get('log_file', '-')}`",
            ]
        )
        if autokaggle.get("valid_output_issues"):
            lines.append(f"- Output issues: `{', '.join(map(str, autokaggle.get('valid_output_issues', [])))}`")
        if autokaggle.get("error"):
            lines.append(f"- Error: `{autokaggle.get('error')}`")
        elif autokaggle.get("stdout_tail"):
            lines.append(f"- Notes: `{str(autokaggle['stdout_tail'])[:240]}`")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    selected = [DATASETS[key] for key in args.datasets] if args.datasets else list(DATASETS.values())
    baseline_rows = load_json(PRIMARY_JSON)
    rows_by_dataset = {row["dataset"]: row for row in baseline_rows}

    for spec in selected:
        metric_name = next(dataset.metric_hint for dataset in ALL_VALIDATION_DATASETS if dataset.key == spec.key)
        result = run_autokaggle(spec, args.env_file, args.timeout_sec, args.valid_size)
        row = dict(rows_by_dataset[spec.key])
        row["autokaggle"] = result
        better, advantage = determine_best_framework(metric_name, row)
        row["better"] = better
        row["advantage"] = advantage
        rows_by_dataset[spec.key] = row

    ordered_rows = [rows_by_dataset[spec.key] for spec in ALL_VALIDATION_DATASETS]
    save_json(PRIMARY_JSON, ordered_rows)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    PRIMARY_MD.write_text(
        render_markdown(
            "# Cross-Framework Matrix Comparison (Augmented With LightAutoDS And AutoKaggle)",
            ordered_rows,
            generated_at,
            english=False,
        ),
        encoding="utf-8",
    )
    ENGLISH_MD.write_text(
        render_markdown(
            "# Cross-Framework Matrix Comparison (English Version, Augmented With AutoKaggle)",
            ordered_rows,
            generated_at,
            english=True,
        ),
        encoding="utf-8",
    )
    print(f"AUTOKAGGLE_MATRIX_JSON::{PRIMARY_JSON}")
    print(f"AUTOKAGGLE_MATRIX_MD::{PRIMARY_MD}")
    print(f"AUTOKAGGLE_MATRIX_EN_MD::{ENGLISH_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
