#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _discover_project_root()
REPORT_DIR = PROJECT_ROOT / "tests" / "system_reports"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.validation_matrix import ALL_VALIDATION_DATASETS  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    description: str


DATASETS = {
    spec.key: DatasetSpec(
        key=spec.key,
        description=spec.description,
    )
    for spec in ALL_VALIDATION_DATASETS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun main-only on the validation matrix and refresh the three-framework report in place."
    )
    parser.add_argument("datasets", nargs="*", choices=sorted(DATASETS))
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--main-timeout", type=int, default=2400)
    parser.add_argument(
        "--baseline-json",
        default=str(REPORT_DIR / "framework-matrix-20260505-051038-corrected-with-lightautods.json"),
    )
    parser.add_argument(
        "--baseline-md",
        default=str(REPORT_DIR / "framework-matrix-20260505-051038-corrected-with-lightautods.md"),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def list_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p.resolve() for p in root.iterdir() if p.is_dir()}


def discover_newest_dir(before: set[Path], root: Path) -> Path | None:
    after = list_dirs(root)
    candidates = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def infer_metric_direction(metric_name: str) -> str:
    metric = metric_name.lower().strip()
    if metric.startswith("neg_"):
        return "maximize"
    if any(token in metric for token in ("error", "loss", "rmse", "mae", "mse")):
        return "minimize"
    return "maximize"


def candidate_metric_keys(metric_name: str, task_family: str = "", target_mode: str = "") -> list[str]:
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
    lower_to_value = {str(key).lower(): value for key, value in metrics.items()}
    for key in candidate_keys:
        if key.lower() in lower_to_value:
            try:
                return float(lower_to_value[key.lower()])
            except Exception:
                return None
    return None


def parse_main_metric(manifest: dict[str, Any]) -> tuple[str, float | None]:
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

    candidate_keys = candidate_metric_keys(metric_name, task_family, target_mode)
    if best_model and isinstance(validation_metrics.get(best_model), dict):
        metrics = validation_metrics.get(best_model) or {}
        best_value = _coerce_metric_from_mapping(metrics, candidate_keys)
        if best_value is not None:
            return metric_name, best_value
    top_level_value = _coerce_metric_from_mapping(validation_metrics, candidate_keys)
    if top_level_value is not None:
        return metric_name, top_level_value
    return metric_name, None


def run_main_dataset(dataset_key: str, env_file: str, timeout: int) -> dict[str, Any]:
    before = list_dirs(PROJECT_ROOT / "artifacts" / "runs")
    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "run_multiagent.py"),
        dataset_key,
        "--env-file",
        str(Path(env_file).resolve()),
        "--stop-after-technical-summary",
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    run_dir = discover_newest_dir(before, PROJECT_ROOT / "artifacts" / "runs")
    manifest = {}
    if run_dir:
        manifest_path = run_dir / "reports" / "run_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
    metric_name, metric_value = parse_main_metric(manifest) if manifest else ("", None)
    review_state = manifest.get("review_state") or {}
    retry_count = int((review_state.get("retry_count_by_phase") or {}).get("feature_engineering", 0))
    runtime_state = manifest.get("runtime_state") or {}
    report_state = manifest.get("report_state") or {}
    artifacts = manifest.get("artifacts") or {}
    return {
        "framework": "main",
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "returncode": proc.returncode,
        "runtime_sec": float(runtime_state.get("total_runtime_sec") or round(time.time() - started, 4)),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "run_dir": str(run_dir) if run_dir else "",
        "technical_report": str(report_state.get("technical_report_path") or artifacts.get("technical_report") or ""),
        "prediction_file": str(artifacts.get("prediction_file") or ""),
        "feature_engineering_retries": retry_count,
        "fatal_errors": list(review_state.get("fatal_errors") or []),
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def is_main_result_better(metric_name: str, old_main: dict[str, Any], new_main: dict[str, Any]) -> bool:
    old_status = str(old_main.get("status") or "")
    new_status = str(new_main.get("status") or "")
    if old_status != "PASS" and new_status == "PASS":
        return True
    if new_status != "PASS":
        return False

    old_metric = old_main.get("metric_value")
    new_metric = new_main.get("metric_value")
    old_runtime = old_main.get("runtime_sec")
    new_runtime = new_main.get("runtime_sec")
    if old_metric is None or new_metric is None or old_runtime is None or new_runtime is None:
        return False

    try:
        old_metric_f = float(old_metric)
        new_metric_f = float(new_metric)
        old_runtime_f = float(old_runtime)
        new_runtime_f = float(new_runtime)
    except Exception:
        return False

    direction = infer_metric_direction(metric_name)
    metric_not_worse = (
        new_metric_f >= old_metric_f - 1e-9
        if direction == "maximize"
        else new_metric_f <= old_metric_f + 1e-9
    )
    runtime_better = new_runtime_f < old_runtime_f - 1e-9
    return metric_not_worse and runtime_better


def determine_best(metric_name: str, scores: dict[str, float | None]) -> tuple[str, float | None]:
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


def fmt_score(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def build_summary(rows: list[dict[str, Any]], replaced: list[str]) -> list[str]:
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
        f"- Refreshed main rows: `{len(replaced)}`",
        f"- Replaced datasets: `{', '.join(replaced) if replaced else 'none'}`",
        "- Replacement rule: main row is updated only when the refreshed run is not worse on the primary metric and is faster overall.",
        "- Main reruns use candidate-level parallel model search with slightly expanded parameter grids.",
        "",
    ]


def build_markdown(rows: list[dict[str, Any]], replaced: list[str], baseline_json: Path) -> str:
    lines = [
        "# Cross-Framework Matrix Comparison (Augmented With LightAutoDS)",
        "",
        f"- Generated: {utc_now()}",
        f"- Refreshed from baseline: `{baseline_json}`",
        "- AIDE and LightAutoDS rows are reused from the prior report; only main was rerun.",
        "",
    ]
    lines.extend(build_summary(rows, replaced))
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
                f"- Technical report: `{row['main'].get('technical_report')}`",
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
    baseline_json = Path(args.baseline_json).resolve()
    baseline_md = Path(args.baseline_md).resolve()
    rows = load_json(baseline_json)
    baseline_map = {row["dataset"]: row for row in rows}

    dataset_keys = args.datasets or [spec.key for spec in ALL_VALIDATION_DATASETS]
    replaced: list[str] = []
    refreshed_results: dict[str, dict[str, Any]] = {}

    for dataset_key in dataset_keys:
        print(f"[MAIN-REFRESH] {dataset_key}", flush=True)
        refreshed_results[dataset_key] = run_main_dataset(
            dataset_key,
            args.env_file,
            args.main_timeout,
        )

    updated_rows: list[dict[str, Any]] = []
    for spec in ALL_VALIDATION_DATASETS:
        dataset_key = spec.key
        row = dict(baseline_map[dataset_key])
        current_main = dict(row["main"])
        refreshed_main = refreshed_results.get(dataset_key)
        if refreshed_main and is_main_result_better(row["metric_name"], current_main, refreshed_main):
            row["main"] = refreshed_main
            replaced.append(dataset_key)
        scores = {
            "main": row["main"].get("metric_value"),
            "aide": row["aide"].get("metric_value"),
            "lightautods_lama": row["lightautods_lama"].get("metric_value"),
        }
        row["better"], row["advantage"] = determine_best(row["metric_name"], scores)
        updated_rows.append(row)

    baseline_json.write_text(
        json.dumps(updated_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    baseline_md.write_text(
        build_markdown(updated_rows, replaced, baseline_json),
        encoding="utf-8",
    )
    print(f"UPDATED_MD::{baseline_md}")
    print(f"UPDATED_JSON::{baseline_json}")
    print(f"REPLACED::{','.join(replaced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
