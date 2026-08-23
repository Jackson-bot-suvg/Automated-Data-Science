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
LADS_ROOT = PROJECT_ROOT / "_external" / "lightautods_eval"
LAMA_PYTHON = LADS_ROOT / "lama_py310" / "bin" / "python"
LAMA_HELPER = PROJECT_ROOT / "tests" / "helpers" / "run_lads_lama_experiment.py"

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
        description="Reuse the corrected main/AIDE matrix baseline and append LightAutoDS LightAutoML-only results."
    )
    parser.add_argument("datasets", nargs="*", choices=sorted(DATASETS))
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--use-parser", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_baseline_json(explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found: {path}")
        return path
    candidates = sorted(REPORT_DIR.glob("framework-matrix-*-corrected.json"))
    if not candidates:
        raise FileNotFoundError("No corrected two-framework baseline JSON found.")
    return candidates[-1]


def build_output_paths(baseline_json: Path) -> tuple[Path, Path]:
    md_path = baseline_json.with_name(baseline_json.stem.replace(".json", "") + "-with-lightautods.md")
    json_path = baseline_json.with_name(baseline_json.stem + "-with-lightautods.json")
    return md_path, json_path


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


def determine_best_framework(metric_name: str, scores: dict[str, float | None]) -> tuple[str, float | None]:
    available = {name: value for name, value in scores.items() if value is not None}
    if not available:
        return "not_comparable", None
    ordered = sorted(
        available.items(),
        key=lambda item: item[1],
        reverse=(infer_metric_direction(metric_name) == "maximize"),
    )
    winner, winner_score = ordered[0]
    if len(ordered) == 1:
        return winner, None
    return winner, abs(float(winner_score) - float(ordered[1][1]))


def run_lama(spec: DatasetSpec, metric_name: str, env_file: str, use_parser: bool) -> dict[str, Any]:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_json = TMP_DIR / f"lads-lama-{spec.key}-{timestamp}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(LAMA_PYTHON),
        str(LAMA_HELPER),
        "--repo-root",
        str((LADS_ROOT / "LADS").resolve()),
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
    ]
    if not use_parser:
        cmd.append("--no-use-parser")
    proc = subprocess.run(
        cmd,
        cwd=LADS_ROOT,
        text=True,
        capture_output=True,
    )
    payload = load_json(output_json)
    return {
        "framework": "lightautods_lama",
        "status": payload.get("status", "FAIL"),
        "returncode": proc.returncode,
        "runtime_sec": payload.get("runtime_sec"),
        "metric_name": payload.get("metric_name", metric_name),
        "metric_value": payload.get("metric_value"),
        "run_dir": payload.get("run_dir", ""),
        "prediction_file": payload.get("prediction_file", ""),
        "prediction_exists": payload.get("prediction_exists", False),
        "validation_predictions_file": payload.get("validation_predictions_file", ""),
        "validation_predictions_exists": payload.get("validation_predictions_exists", False),
        "metrics_json": payload.get("metrics_json", ""),
        "route": payload.get("route", "LAMA"),
        "internal_config": payload.get("internal_config", {}),
        "parser_response": payload.get("parser_response", ""),
        "stdout_tail": payload.get("stdout_tail", ""),
        "stderr_tail": payload.get("stderr_tail", ""),
        "error": payload.get("error", ""),
    }


def build_summary(rows: list[dict[str, Any]]) -> list[str]:
    comparable = [row for row in rows if row["better"] != "not_comparable"]
    main_wins = sum(1 for row in comparable if row["better"] == "main")
    aide_wins = sum(1 for row in comparable if row["better"] == "aide")
    lads_wins = sum(1 for row in comparable if row["better"] == "lightautods_lama")
    main_runtime = sum(float((row["main"] or {}).get("runtime_sec") or 0.0) for row in rows)
    aide_runtime = sum(float((row["aide"] or {}).get("runtime_sec") or 0.0) for row in rows)
    lads_runtime = sum(float((row["lightautods_lama"] or {}).get("runtime_sec") or 0.0) for row in rows)
    return [
        "## Snapshot",
        "",
        f"- Comparable tasks: `{len(comparable)}/{len(rows)}`",
        f"- Win count: `main={main_wins}`, `aide={aide_wins}`, `lightautods_lama={lads_wins}`",
        f"- Total runtime: `main={main_runtime:.4f}s`, `aide={aide_runtime:.4f}s`, `lightautods_lama={lads_runtime:.4f}s`",
        "- LightAutoDS route: `LightAutoML-only` with `MiniMax 2.7` parser-selected config, then external scoring on the main-project metric.",
        "- Compatibility note: LightAutoDS training is forced to a single-process LightAutoML preset for stability on this machine.",
        "",
    ]


def build_markdown(rows: list[dict[str, Any]], baseline_json: Path, use_parser: bool) -> str:
    lines = [
        "# Cross-Framework Matrix Comparison (Augmented With LightAutoDS)",
        "",
        f"- Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Baseline source: `{baseline_json}`",
        f"- LightAutoDS parser mode: `{'enabled' if use_parser else 'disabled'}`",
        "- Main and AIDE rows are copied from the existing corrected baseline; only LightAutoDS was rerun.",
        "",
    ]
    lines.extend(build_summary(rows))
    lines.extend(
        [
            "## Summary",
            "",
            "| Dataset | Primary Metric | Main | AIDE | LightAutoDS | Better | Advantage | Main Time (s) | AIDE Time (s) | LightAutoDS Time (s) |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| `{dataset}` | `{metric}` | {main} | {aide} | {lads} | `{better}` | {adv} | {main_t} | {aide_t} | {lads_t} |".format(
                dataset=row["dataset"],
                metric=row["metric_name"],
                main=fmt_score(row["main"].get("metric_value")),
                aide=fmt_score(row["aide"].get("metric_value")),
                lads=fmt_score(row["lightautods_lama"].get("metric_value")),
                better=row["better"],
                adv=fmt_score(row["advantage"]),
                main_t=fmt_score(row["main"].get("runtime_sec")),
                aide_t=fmt_score(row["aide"].get("runtime_sec")),
                lads_t=fmt_score(row["lightautods_lama"].get("runtime_sec")),
            )
        )
    lines.append("")

    for idx, row in enumerate(rows, start=1):
        lads = row["lightautods_lama"]
        lines.extend(
            [
                f"## {idx}. `{row['dataset']}`",
                "",
                f"- Description: `{row['description']}`",
                f"- Primary metric: `{row['metric_name']}`",
                f"- Better framework: `{row['better']}`",
                f"- Metric advantage: `{fmt_score(row['advantage'])}`",
                "",
                "### Main",
                f"- Status: `{row['main'].get('status')}`",
                f"- Metric: `{fmt_score(row['main'].get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(row['main'].get('runtime_sec'))}`",
                f"- Prediction file: `{row['main'].get('prediction_file')}`",
                "",
                "### AIDE",
                f"- Status: `{row['aide'].get('status')}`",
                f"- Metric: `{fmt_score(row['aide'].get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(row['aide'].get('runtime_sec'))}`",
                f"- Prediction file: `{row['aide'].get('prediction_file')}`",
                "",
                "### LightAutoDS",
                f"- Status: `{lads.get('status')}`",
                f"- Metric: `{fmt_score(lads.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(lads.get('runtime_sec'))}`",
                f"- Route: `{lads.get('route')}`",
                f"- Internal config: `{json.dumps(lads.get('internal_config') or {}, ensure_ascii=False)}`",
                f"- Validation predictions: `{lads.get('validation_predictions_file')}`",
                f"- Prediction file: `{lads.get('prediction_file')}`",
            ]
        )
        if lads.get("error"):
            lines.append(f"- Error: `{lads.get('error')}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    baseline_path = discover_baseline_json(args.baseline_json)
    baseline_rows = load_json(baseline_path)
    baseline_map = {row["dataset"]: row for row in baseline_rows}

    dataset_keys = args.datasets or [row["dataset"] for row in baseline_rows]
    rows: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        spec = DATASETS.get(dataset_key)
        baseline = baseline_map.get(dataset_key)
        if not spec or not baseline:
            raise KeyError(f"Dataset missing in spec or baseline: {dataset_key}")
        metric_name = str(baseline["metric_name"])
        lama_result = run_lama(spec, metric_name, args.env_file, args.use_parser)
        score_map = {
            "main": baseline["main"].get("metric_value"),
            "aide": baseline["aide"].get("metric_value"),
            "lightautods_lama": lama_result.get("metric_value"),
        }
        better, advantage = determine_best_framework(metric_name, score_map)
        rows.append(
            {
                "dataset": dataset_key,
                "description": spec.description,
                "metric_name": metric_name,
                "better": better,
                "advantage": advantage,
                "main": baseline["main"],
                "aide": baseline["aide"],
                "lightautods_lama": lama_result,
            }
        )

    md_path, json_path = build_output_paths(baseline_path)
    md_path.write_text(build_markdown(rows, baseline_path, args.use_parser), encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"LAMA_AUGMENTED_MD::{md_path}")
    print(f"LAMA_AUGMENTED_JSON::{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
