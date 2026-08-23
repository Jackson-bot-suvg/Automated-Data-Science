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
AIDE_ROOT = PROJECT_ROOT / "_external" / "aide_eval" / "aideml"
AIDE_HELPER = PROJECT_ROOT / "tests" / "helpers" / "run_aide_experiment.py"

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
    parser = argparse.ArgumentParser(
        description="Run the main workflow and AIDE across the validation matrix and write a comparison report."
    )
    parser.add_argument("datasets", nargs="*", choices=sorted(PRESETS))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--main-timeout", type=int, default=2400)
    parser.add_argument("--aide-timeout", type=int, default=1800)
    parser.add_argument("--aide-steps", type=int, default=2)
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


def build_report_paths() -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    md = REPORT_DIR / f"framework-matrix-{stamp}.md"
    js = REPORT_DIR / f"framework-matrix-{stamp}.json"
    return md, js


def list_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p.resolve() for p in root.iterdir() if p.is_dir()}


def discover_newest_dir(before: set[Path], root: Path) -> Path | None:
    after = list_dirs(root)
    candidates = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.datasets:
        return [PRESETS[key] for key in args.datasets]
    return [PRESETS[spec.key] for spec in ALL_VALIDATION_DATASETS]


def strip_provider_prefix(model_name: str) -> str:
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def infer_metric_direction(metric_name: str) -> str:
    metric = metric_name.lower().strip()
    if metric.startswith("neg_"):
        return "maximize"
    if any(token in metric for token in ("error", "loss", "rmse", "mae", "mse")):
        return "minimize"
    return "maximize"


def build_aide_goal(spec: DatasetSpec) -> str:
    target = spec.target.replace(",", ", ")
    if spec.target_mode == "multi_label":
        return (
            f"{spec.task.capitalize()} for target columns {target}. "
            "Build a machine learning pipeline that predicts all required outputs."
        )
    return f"{spec.task.capitalize()} for target column {target}."


def build_aide_eval_prompt(spec: DatasetSpec, primary_metric: str) -> str:
    target = spec.target.replace(",", ", ")
    metric = primary_metric.strip()
    if metric == "accuracy":
        return (
            f"Use the main project's primary metric `{metric}`. "
            "Evaluate with cross-validation or a hold-out validation set and print validation accuracy."
        )
    if metric == "roc_auc":
        return (
            f"Use the main project's primary metric `{metric}`. "
            "Evaluate with ROC AUC on validation data using predicted probabilities and print the metric value."
        )
    if metric == "f1_macro":
        return (
            f"Use the main project's primary metric `{metric}`. "
            "Evaluate with macro-averaged F1 on validation data and print the metric value."
        )
    if metric == "roc_auc_macro_ovr":
        return (
            f"Use the main project's primary metric `{metric}` for multilabel targets {target}. "
            "Compute macro-averaged one-vs-rest ROC AUC from validation predicted probabilities and print the metric value."
        )
    if metric == "neg_root_mean_squared_error":
        return (
            f"Use the main project's primary metric `{metric}`. "
            "Evaluate with cross-validation scoring='neg_root_mean_squared_error' and print that value "
            "(higher is better, meaning less negative is better)."
        )
    return (
        f"Use the main project's primary metric `{metric}`. "
        "Evaluate on validation data and print the metric value clearly."
    )


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
    evaluation = (target_spec.get("evaluation") or {})
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


def run_main_dataset(spec: DatasetSpec, env_file: str, timeout: int) -> dict[str, Any]:
    before = list_dirs(PROJECT_ROOT / "artifacts" / "runs")
    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "run_multiagent.py"),
        spec.key,
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


def build_aide_env(env_file: str) -> tuple[dict[str, str], str, str]:
    values = dotenv_values(env_file)
    base_url = str(values.get("LLM_MINIMAX_BASE_URL") or values.get("OPENAI_BASE_URL") or "").strip()
    api_key = str(values.get("LLM_MINIMAX_API_KEY") or values.get("OPENAI_API_KEY") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError("MiniMax/OpenAI-compatible API credentials are missing from env file.")
    code_model = strip_provider_prefix(str(values.get("AGENT_MODEL_BUILDING_MODEL") or "MiniMax-M2.7"))
    feedback_model = strip_provider_prefix(str(values.get("AGENT_REVIEWER_MODEL") or code_model))

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_KEY"] = api_key
    return env, code_model, feedback_model


def run_aide_dataset(
    spec: DatasetSpec,
    env_file: str,
    timeout: int,
    steps: int,
    primary_metric: str,
) -> dict[str, Any]:
    env, code_model, feedback_model = build_aide_env(env_file)
    before_logs = list_dirs(AIDE_ROOT / "logs")
    output_json = REPORT_DIR / "tmp" / f"aide-{spec.key}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(AIDE_ROOT / ".venv" / "bin" / "python"),
        str(AIDE_HELPER),
        "--data-dir",
        str((PROJECT_ROOT / spec.dataset).resolve()),
        "--goal",
        build_aide_goal(spec),
        "--eval",
        build_aide_eval_prompt(spec, primary_metric),
        "--code-model",
        code_model,
        "--feedback-model",
        feedback_model,
        "--steps",
        str(steps),
        "--num-drafts",
        "1",
        "--debug-prob",
        "0.0",
        "--output-json",
        str(output_json),
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    payload = load_json(output_json) if output_json.exists() else {}
    log_dir = discover_newest_dir(before_logs, AIDE_ROOT / "logs")
    return {
        "framework": "aide",
        "status": payload.get("status", "FAIL"),
        "returncode": proc.returncode,
        "runtime_sec": float(payload.get("runtime_sec") or round(time.time() - started, 4)),
        "metric_name": primary_metric,
        "metric_value": payload.get("metric_value"),
        "run_dir": str(log_dir or payload.get("log_dir") or ""),
        "prediction_file": str(payload.get("submission_path") or ""),
        "submission_exists": bool(payload.get("submission_exists")),
        "best_solution_path": str(payload.get("best_solution_path") or ""),
        "best_solution_loc": int(payload.get("best_solution_loc") or 0),
        "nodes_total": int(payload.get("nodes_total") or 0),
        "buggy_nodes": int(payload.get("buggy_nodes") or 0),
        "good_nodes": int(payload.get("good_nodes") or 0),
        "analysis": str(payload.get("best_node_analysis") or ""),
        "error": str(payload.get("error") or ""),
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def compare_scores(metric_name: str, main_score: float | None, aide_score: float | None) -> tuple[str, float | None]:
    if main_score is None or aide_score is None:
        return "not_comparable", None
    direction = infer_metric_direction(metric_name)
    delta = main_score - aide_score
    if direction == "maximize":
        if delta > 0:
            return "main", delta
        if delta < 0:
            return "aide", abs(delta)
        return "tie", 0.0
    if delta < 0:
        return "main", abs(delta)
    if delta > 0:
        return "aide", delta
    return "tie", 0.0


def fmt_score(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def build_summary_stats(results: list[dict[str, Any]]) -> list[str]:
    comparable = [row for row in results if row.get("better") not in {"not_comparable", ""}]
    main_wins = sum(1 for row in comparable if row.get("better") == "main")
    aide_wins = sum(1 for row in comparable if row.get("better") == "aide")
    ties = sum(1 for row in comparable if row.get("better") == "tie")
    main_total_time = sum(float((row.get("main") or {}).get("runtime_sec") or 0.0) for row in results)
    aide_total_time = sum(float((row.get("aide") or {}).get("runtime_sec") or 0.0) for row in results)
    main_retries = sum(int((row.get("main") or {}).get("feature_engineering_retries") or 0) for row in results)
    aide_buggy_nodes = sum(int((row.get("aide") or {}).get("buggy_nodes") or 0) for row in results)
    return [
        "## Snapshot",
        "",
        f"- Comparable tasks so far: `{len(comparable)}/{len(results)}`",
        f"- Win count: `main={main_wins}`, `aide={aide_wins}`, `tie={ties}`",
        f"- Total runtime so far: `main={fmt_score(main_total_time)}s`, `aide={fmt_score(aide_total_time)}s`",
        f"- Stability signals so far: `main_feature_retries={main_retries}`, `aide_buggy_nodes={aide_buggy_nodes}`",
        "",
    ]


def write_report(report_path: Path, json_path: Path, results: list[dict[str, Any]], datasets: list[DatasetSpec], args: argparse.Namespace) -> None:
    lines = [
        "# Cross-Framework Matrix Comparison",
        "",
        f"- Generated: {utc_now()}",
        f"- Commit: `{git_commit()}`",
        f"- Dataset count: `{len(datasets)}`",
        f"- Main timeout: `{args.main_timeout}s`",
        f"- AIDE timeout: `{args.aide_timeout}s`",
        f"- AIDE steps: `{args.aide_steps}`",
        "",
    ]
    if results:
        lines.extend(build_summary_stats(results))
    lines.extend(
        [
        "## Summary",
        "",
        "| Dataset | Primary Metric | Main | AIDE | Better | Advantage | Main Time (s) | AIDE Time (s) | Main FE Retries | AIDE Buggy Nodes |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in results:
        lines.append(
            f"| `{row['dataset']}` | `{row['metric_name'] or '-'}` | {fmt_score(row['main'].get('metric_value'))} | "
            f"{fmt_score(row['aide'].get('metric_value'))} | `{row['better']}` | {fmt_score(row['advantage'])} | "
            f"{fmt_score(row['main'].get('runtime_sec'))} | {fmt_score(row['aide'].get('runtime_sec'))} | "
            f"{row['main'].get('feature_engineering_retries', 0)} | {row['aide'].get('buggy_nodes', 0)} |"
        )

    for idx, row in enumerate(results, start=1):
        main = row["main"]
        aide = row["aide"]
        lines.extend(
            [
                "",
                f"## {idx}. `{row['dataset']}`",
                "",
                f"- Description: `{row['description']}`",
                f"- Primary metric: `{row['metric_name'] or '-'}`",
                f"- Better framework: `{row['better']}`",
                f"- Metric advantage: `{fmt_score(row['advantage'])}`",
                "",
                "### Main",
                f"- Status: `{main.get('status')}`",
                f"- Metric: `{fmt_score(main.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(main.get('runtime_sec'))}`",
                f"- Feature-engineering retries: `{main.get('feature_engineering_retries', 0)}`",
                f"- Technical report: `{main.get('technical_report') or '-'}`",
                f"- Prediction file: `{main.get('prediction_file') or '-'}`",
                "",
                "### AIDE",
                f"- Status: `{aide.get('status')}`",
                f"- Metric: `{fmt_score(aide.get('metric_value'))}`",
                f"- Runtime (s): `{fmt_score(aide.get('runtime_sec'))}`",
                f"- Buggy nodes: `{aide.get('buggy_nodes', 0)}` / total nodes `{aide.get('nodes_total', 0)}`",
                f"- Best solution LOC: `{aide.get('best_solution_loc', 0)}`",
                f"- Prediction file: `{aide.get('prediction_file') or '-'}`",
            ]
        )
        if main.get("fatal_errors"):
            lines.append(f"- Main fatal errors: `{'; '.join(main['fatal_errors'])}`")
        if aide.get("error"):
            lines.append(f"- AIDE error: `{aide['error']}`")
        if aide.get("analysis"):
            lines.append(f"- AIDE best-node analysis: {aide['analysis']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets = resolve_datasets(args)
    report_path, json_path = build_report_paths()
    results: list[dict[str, Any]] = []

    for spec in datasets:
        print(f"[COMPARE] {spec.key}", flush=True)
        main_result = run_main_dataset(spec, args.env_file, args.main_timeout)
        metric_name = str(main_result.get("metric_name") or "").strip()
        if not metric_name:
            metric_name = {
                "binary_classification": "accuracy",
                "multiclass_classification": "f1_macro",
                "regression": "neg_root_mean_squared_error",
                "multilabel_classification": "roc_auc_macro_ovr",
            }.get(spec.task_family, "accuracy")
        aide_result = run_aide_dataset(
            spec,
            args.env_file,
            args.aide_timeout,
            args.aide_steps,
            metric_name,
        )
        better, advantage = compare_scores(
            metric_name,
            main_result.get("metric_value"),
            aide_result.get("metric_value"),
        )
        results.append(
            {
                "dataset": spec.key,
                "description": spec.description,
                "metric_name": metric_name,
                "better": better,
                "advantage": advantage,
                "main": main_result,
                "aide": aide_result,
            }
        )
        write_report(report_path, json_path, results, datasets, args)
        print(f"[COMPARE] report updated: {report_path}", flush=True)

    print(f"REPORT::{report_path}")
    print(f"JSON::{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
