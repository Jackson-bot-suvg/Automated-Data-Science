#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values


def _discover_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "run_multiagent.py").exists():
            return parent
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _discover_project_root()
REPORT_DIR = PROJECT_ROOT / "tests" / "system_reports"
TMP_DIR = REPORT_DIR / "tmp"
PROXY_HELPER = PROJECT_ROOT / "tests" / "helpers" / "openai_usage_proxy.py"
AIDE_HELPER = PROJECT_ROOT / "tests" / "helpers" / "run_aide_experiment.py"
LADS_HELPER = PROJECT_ROOT / "tests" / "helpers" / "run_lads_lama_experiment.py"
AUTOKAGGLE_HELPER = PROJECT_ROOT / "tests" / "helpers" / "run_autokaggle_experiment.py"
AIDE_ROOT = PROJECT_ROOT / "_external" / "aide_eval" / "aideml"
AUTOKAGGLE_ROOT = PROJECT_ROOT / "_external" / "autokaggle_eval" / "AutoKaggle"

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
    metric_name: str


DATASETS = {
    spec.key: DatasetSpec(
        key=spec.key,
        task=spec.task,
        dataset=spec.dataset,
        target=spec.target,
        description=spec.description,
        task_family=spec.task_family,
        target_mode=spec.target_mode,
        metric_name=spec.metric_hint,
    )
    for spec in ALL_VALIDATION_DATASETS
}

FRAMEWORKS = ("main", "aide", "lightautods_lama", "autokaggle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fresh multi-framework matrix with repeatability and token usage metrics."
    )
    parser.add_argument("datasets", nargs="*", choices=sorted(DATASETS))
    parser.add_argument("--frameworks", nargs="*", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--main-timeout", type=int, default=2400)
    parser.add_argument("--aide-timeout", type=int, default=1800)
    parser.add_argument("--lads-timeout", type=int, default=900)
    parser.add_argument("--autokaggle-timeout", type=int, default=2400)
    parser.add_argument("--aide-steps", type=int, default=2)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--output-stem", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing raw JSONL run log for the same output stem and skip completed jobs.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def infer_metric_direction(metric_name: str) -> str:
    metric = metric_name.lower().strip()
    if metric.startswith("neg_"):
        return "maximize"
    if any(token in metric for token in ("error", "loss", "rmse", "mae", "mse")):
        return "minimize"
    return "maximize"


def fmt_score(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def strip_provider_prefix(model_name: str) -> str:
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def build_aide_goal(spec: DatasetSpec) -> str:
    target = spec.target.replace(",", ", ")
    if spec.target_mode == "multi_label":
        return (
            f"{spec.task.capitalize()} for target columns {target}. "
            "Build a machine learning pipeline that predicts all required outputs."
        )
    return f"{spec.task.capitalize()} for target column {target}."


def build_aide_eval_prompt(spec: DatasetSpec) -> str:
    target = spec.target.replace(",", ", ")
    metric = spec.metric_name.strip()
    if metric == "accuracy":
        return "Evaluate with validation accuracy and print the metric value."
    if metric == "roc_auc":
        return "Evaluate ROC AUC on validation data using predicted probabilities and print the metric value."
    if metric == "f1_macro":
        return "Evaluate macro-averaged F1 on validation data and print the metric value."
    if metric == "roc_auc_macro_ovr":
        return (
            f"Use macro-averaged one-vs-rest ROC AUC for multilabel targets {target}. "
            "Compute it from validation predicted probabilities and print the metric value."
        )
    if metric == "neg_root_mean_squared_error":
        return (
            "Evaluate with cross-validation scoring='neg_root_mean_squared_error' and print that value "
            "(higher is better, meaning less negative is better)."
        )
    return f"Evaluate on validation data using `{metric}` and print the metric value clearly."


def candidate_metric_keys(metric_name: str, task_family: str = "", target_mode: str = "") -> list[str]:
    keys = [metric_name, f"cv_{metric_name}_mean", f"val_{metric_name}", f"{metric_name}_mean"]
    aliases = {
        "roc_auc_macro_ovr": [
            "cv_per_target_auc_mean",
            "per_target_auc_mean",
            "cv_roc_auc_macro_ovr_mean",
            "roc_auc_macro_ovr_mean",
        ],
        "f1_macro": ["cv_macro_f1_mean", "macro_f1_mean"],
        "neg_root_mean_squared_error": ["cv_rmse_mean", "rmse_mean"],
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
        best_value = _coerce_metric_from_mapping(validation_metrics.get(best_model) or {}, candidate_keys)
        if best_value is not None:
            return metric_name, best_value
    top_level_value = _coerce_metric_from_mapping(validation_metrics, candidate_keys)
    return metric_name, top_level_value


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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_env_values(env_file: str) -> dict[str, str]:
    values = dotenv_values(env_file)
    return {str(key): str(value) for key, value in values.items() if value is not None}


def resolve_api_env(env_values: dict[str, str]) -> tuple[str, str]:
    base_url = (
        env_values.get("LLM_MINIMAX_BASE_URL")
        or env_values.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_MINIMAX_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip()
    api_key = (
        env_values.get("LLM_MINIMAX_API_KEY")
        or env_values.get("OPENAI_API_KEY")
        or os.environ.get("LLM_MINIMAX_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not base_url or not api_key:
        raise RuntimeError("Missing MiniMax/OpenAI-compatible base URL or API key.")
    return base_url, api_key


def write_temp_env(base_env: dict[str, str], *, base_url: str, api_key: str) -> str:
    payload = dict(base_env)
    payload["OPENAI_BASE_URL"] = base_url
    payload["OPENAI_API_KEY"] = api_key
    payload["LLM_MINIMAX_BASE_URL"] = base_url
    payload["LLM_MINIMAX_API_KEY"] = api_key
    fd, temp_path = tempfile.mkstemp(prefix="framework-matrix-", suffix=".env")
    os.close(fd)
    path = Path(temp_path)
    lines = [f"{key}={value}" for key, value in sorted(payload.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class UsageProxy:
    def __init__(self, *, upstream_base_url: str, log_path: Path) -> None:
        self.upstream_base_url = upstream_base_url
        self.log_path = log_path
        self.port = find_free_port()
        self.process: subprocess.Popen[str] | None = None

    @property
    def proxy_base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():
            self.log_path.unlink()
        cmd = [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(PROXY_HELPER),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--upstream-base-url",
            self.upstream_base_url,
            "--log-path",
            str(self.log_path),
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.proxy_base_url}/health", timeout=1)
                if resp.ok:
                    return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("Usage proxy failed to start within 15 seconds.")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None


def parse_usage_log(path: Path) -> dict[str, Any]:
    metrics = {
        "input_token_cost": 0,
        "output_token_cost": 0,
        "token_total": 0,
        "llm_call_count": 0,
    }
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        metrics["llm_call_count"] += 1
        metrics["input_token_cost"] += int(record.get("input_tokens") or 0)
        metrics["output_token_cost"] += int(record.get("output_tokens") or 0)
        metrics["token_total"] += int(record.get("total_tokens") or 0)
    return metrics


def run_main_dataset(spec: DatasetSpec, env_file: str, timeout: int, usage_log: Path) -> dict[str, Any]:
    base_env = load_env_values(env_file)
    upstream_base_url, api_key = resolve_api_env(base_env)
    proxy = UsageProxy(upstream_base_url=upstream_base_url, log_path=usage_log)
    before = list_dirs(PROJECT_ROOT / "artifacts" / "runs")
    temp_env = None
    started = time.time()
    try:
        proxy.start()
        temp_env = write_temp_env(base_env, base_url=proxy.proxy_base_url, api_key=api_key)
        cmd = [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(PROJECT_ROOT / "run_multiagent.py"),
            spec.key,
            "--env-file",
            temp_env,
            "--stop-after-technical-summary",
        ]
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    finally:
        proxy.stop()
        if temp_env and Path(temp_env).exists():
            Path(temp_env).unlink()

    run_dir = discover_newest_dir(before, PROJECT_ROOT / "artifacts" / "runs")
    manifest = {}
    if run_dir:
        manifest_path = run_dir / "reports" / "run_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
    metric_name, metric_value = parse_main_metric(manifest) if manifest else (spec.metric_name, None)
    runtime_state = manifest.get("runtime_state") or {}
    artifacts = manifest.get("artifacts") or {}
    review_state = manifest.get("review_state") or {}
    prediction_file = str(artifacts.get("prediction_file") or "")
    usage = parse_usage_log(usage_log)
    return {
        "framework": "main",
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "metric_name": metric_name or spec.metric_name,
        "metric_value": metric_value,
        "runtime_sec": float(runtime_state.get("total_runtime_sec") or round(time.time() - started, 4)),
        "valid_output": bool(prediction_file and Path(prediction_file).exists()),
        "prediction_file": prediction_file,
        "run_dir": str(run_dir or ""),
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "usage_log": str(usage_log),
        **usage,
        "fatal_errors": list(review_state.get("fatal_errors") or []),
    }


def run_aide_dataset(spec: DatasetSpec, env_file: str, timeout: int, steps: int, usage_log: Path) -> dict[str, Any]:
    env_values = load_env_values(env_file)
    upstream_base_url, api_key = resolve_api_env(env_values)
    proxy = UsageProxy(upstream_base_url=upstream_base_url, log_path=usage_log)
    before_logs = list_dirs(AIDE_ROOT / "logs")
    started = time.time()
    output_json = TMP_DIR / f"aide-{spec.key}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}.json"
    code_model = strip_provider_prefix(str(env_values.get("AGENT_MODEL_BUILDING_MODEL") or "MiniMax-M2.7"))
    feedback_model = strip_provider_prefix(str(env_values.get("AGENT_REVIEWER_MODEL") or code_model))
    proc = None
    try:
        proxy.start()
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = proxy.proxy_base_url
        env["OPENAI_API_KEY"] = api_key
        cmd = [
            str(AIDE_ROOT / ".venv" / "bin" / "python"),
            str(AIDE_HELPER),
            "--data-dir",
            str((PROJECT_ROOT / spec.dataset).resolve()),
            "--goal",
            build_aide_goal(spec),
            "--eval",
            build_aide_eval_prompt(spec),
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
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    finally:
        proxy.stop()

    payload = load_json(output_json) if output_json.exists() else {}
    usage = parse_usage_log(usage_log)
    log_dir = discover_newest_dir(before_logs, AIDE_ROOT / "logs")
    return {
        "framework": "aide",
        "status": payload.get("status", "FAIL"),
        "metric_name": spec.metric_name,
        "metric_value": payload.get("metric_value"),
        "runtime_sec": float(payload.get("runtime_sec") or round(time.time() - started, 4)),
        "valid_output": bool(payload.get("submission_exists")),
        "prediction_file": str(payload.get("submission_path") or ""),
        "run_dir": str(log_dir or payload.get("log_dir") or ""),
        "stdout_tail": ((proc.stdout if proc else "") or "")[-4000:],
        "stderr_tail": ((proc.stderr if proc else "") or "")[-2000:],
        "usage_log": str(usage_log),
        **usage,
        "nodes_total": int(payload.get("nodes_total") or 0),
        "buggy_nodes": int(payload.get("buggy_nodes") or 0),
        "good_nodes": int(payload.get("good_nodes") or 0),
        "error": str(payload.get("error") or ""),
    }


def run_lads_dataset(spec: DatasetSpec, env_file: str, timeout: int, usage_log: Path) -> dict[str, Any]:
    base_env = load_env_values(env_file)
    upstream_base_url, api_key = resolve_api_env(base_env)
    proxy = UsageProxy(upstream_base_url=upstream_base_url, log_path=usage_log)
    temp_env = None
    output_json = TMP_DIR / f"lads-lama-{spec.key}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}.json"
    started = time.time()
    proc = None
    try:
        proxy.start()
        temp_env = write_temp_env(base_env, base_url=proxy.proxy_base_url, api_key=api_key)
        cmd = [
            str(PROJECT_ROOT / "_external" / "lightautods_eval" / "lama_py310" / "bin" / "python"),
            str(LADS_HELPER),
            "--repo-root",
            str((PROJECT_ROOT / "_external" / "lightautods_eval" / "LADS").resolve()),
            "--data-dir",
            str((PROJECT_ROOT / spec.dataset).resolve()),
            "--task",
            spec.task,
            "--target",
            spec.target,
            "--metric-name",
            spec.metric_name,
            "--task-family",
            spec.task_family,
            "--description",
            spec.description,
            "--output-json",
            str(output_json),
            "--env-file",
            temp_env,
        ]
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    finally:
        proxy.stop()
        if temp_env and Path(temp_env).exists():
            Path(temp_env).unlink()

    payload = load_json(output_json) if output_json.exists() else {}
    usage = parse_usage_log(usage_log)
    return {
        "framework": "lightautods_lama",
        "status": payload.get("status", "FAIL"),
        "metric_name": payload.get("metric_name", spec.metric_name),
        "metric_value": payload.get("metric_value"),
        "runtime_sec": float(payload.get("runtime_sec") or round(time.time() - started, 4)),
        "valid_output": bool(payload.get("prediction_exists")),
        "prediction_file": str(payload.get("prediction_file") or ""),
        "run_dir": str(payload.get("run_dir") or ""),
        "stdout_tail": ((proc.stdout if proc else "") or "")[-4000:],
        "stderr_tail": ((proc.stderr if proc else "") or "")[-2000:],
        "usage_log": str(usage_log),
        **usage,
        "internal_config": payload.get("internal_config") or {},
        "error": str(payload.get("error") or ""),
    }


def run_autokaggle_dataset(spec: DatasetSpec, env_file: str, timeout: int, valid_size: float, usage_log: Path) -> dict[str, Any]:
    base_env = load_env_values(env_file)
    upstream_base_url, api_key = resolve_api_env(base_env)
    proxy = UsageProxy(upstream_base_url=upstream_base_url, log_path=usage_log)
    temp_env = None
    output_json = TMP_DIR / f"autokaggle-{spec.key}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}.json"
    started = time.time()
    proc = None
    try:
        proxy.start()
        temp_env = write_temp_env(base_env, base_url=proxy.proxy_base_url, api_key=api_key)
        cmd = [
            str(PROJECT_ROOT / "_external" / "autokaggle_eval" / ".venv" / "bin" / "python"),
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
            spec.metric_name,
            "--task-family",
            spec.task_family,
            "--description",
            spec.description,
            "--output-json",
            str(output_json),
            "--env-file",
            temp_env,
            "--timeout-sec",
            str(timeout),
            "--valid-size",
            str(valid_size),
        ]
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout + 60,
        )
    finally:
        proxy.stop()
        if temp_env and Path(temp_env).exists():
            Path(temp_env).unlink()

    payload = load_json(output_json) if output_json.exists() else {}
    usage = parse_usage_log(usage_log)
    return {
        "framework": "autokaggle",
        "status": payload.get("status", "FAIL"),
        "metric_name": payload.get("metric_name", spec.metric_name),
        "metric_value": payload.get("metric_value"),
        "runtime_sec": float(payload.get("runtime_sec") or round(time.time() - started, 4)),
        "valid_output": bool(payload.get("valid_output")),
        "prediction_file": str(payload.get("prediction_file") or ""),
        "run_dir": str(payload.get("competition_dir") or ""),
        "stdout_tail": ((proc.stdout if proc else "") or "")[-4000:],
        "stderr_tail": ((proc.stderr if proc else "") or "")[-2000:],
        "usage_log": str(usage_log),
        **usage,
        "error": str(payload.get("error") or ""),
    }


def aggregate_runs(dataset: str, framework: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_values = [float(run["metric_value"]) for run in runs if run.get("metric_value") is not None]
    runtimes = [float(run["runtime_sec"]) for run in runs if run.get("runtime_sec") is not None]
    input_tokens = [int(run.get("input_token_cost") or 0) for run in runs]
    output_tokens = [int(run.get("output_token_cost") or 0) for run in runs]
    total_tokens = [int(run.get("token_total") or 0) for run in runs]
    llm_calls = [int(run.get("llm_call_count") or 0) for run in runs]
    success_count = sum(1 for run in runs if run.get("status") == "PASS")
    valid_output_count = sum(1 for run in runs if run.get("valid_output"))

    representative = next((run for run in runs if run.get("status") == "PASS"), runs[0] if runs else {})
    metric_name = str(representative.get("metric_name", ""))
    metric_mean = statistics.mean(metric_values) if metric_values else None
    runtime_mean = statistics.mean(runtimes) if runtimes else None
    reproducibility_std = statistics.stdev(metric_values) if len(metric_values) >= 2 else 0.0 if len(metric_values) == 1 else None
    runtime_std = statistics.stdev(runtimes) if len(runtimes) >= 2 else 0.0 if len(runtimes) == 1 else None
    metric_direction = infer_metric_direction(metric_name) if metric_name else "maximize"
    if metric_values:
        top_score = max(metric_values) if metric_direction == "maximize" else min(metric_values)
    else:
        top_score = None

    return {
        "dataset": dataset,
        "framework": framework,
        "metric_name": metric_name,
        "score_mean": metric_mean,
        "top_score": top_score,
        "reproducibility_std": reproducibility_std,
        "runtime_mean_sec": runtime_mean,
        "runtime_std": runtime_std,
        "input_token_cost": round(statistics.mean(input_tokens), 4) if input_tokens else 0.0,
        "output_token_cost": round(statistics.mean(output_tokens), 4) if output_tokens else 0.0,
        "token_total": round(statistics.mean(total_tokens), 4) if total_tokens else 0.0,
        "llm_call_count": round(statistics.mean(llm_calls), 4) if llm_calls else 0.0,
        "success_rate": f"{success_count}/{len(runs)}",
        "valid_output_rate": f"{valid_output_count}/{len(runs)}",
        "prediction_file": representative.get("prediction_file", ""),
        "run_dir": representative.get("run_dir", ""),
        "runs": runs,
    }


def determine_best_framework(metric_name: str, framework_rows: dict[str, dict[str, Any]], score_field: str = "score_mean") -> tuple[str, float | None]:
    scores: dict[str, float] = {}
    for framework, row in framework_rows.items():
        if row.get(score_field) is None:
            continue
        scores[framework] = float(row[score_field])
    if not scores:
        return "not_comparable", None
    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=(infer_metric_direction(metric_name) == "maximize"),
    )
    winner, best_value = ordered[0]
    if len(ordered) == 1:
        return winner, None
    return winner, abs(float(best_value) - float(ordered[1][1]))


def render_markdown(
    report_title: str,
    datasets: list[DatasetSpec],
    aggregated: dict[str, dict[str, dict[str, Any]]],
    generated_at: str,
    *,
    score_field: str = "score_mean",
    score_label: str = "Score Mean",
) -> str:
    lines = [
        report_title,
        "",
        f"- Generated: {generated_at}",
        "- Reference baseline document: [`framework-matrix-20260505-051038-corrected-with-lightautods.md`](/Users/a99/Downloads/5703-CS33-2/tests/system_reports/framework-matrix-20260505-051038-corrected-with-lightautods.md)",
        "- This report is a fresh rerun with repeatability and token-usage instrumentation.",
        "- Metric fields use 3-run aggregates per framework-task unless explicitly noted otherwise.",
        "",
        "## Dataset Winners",
        "",
        "| Dataset | Primary Metric | Better | Advantage |",
        "| --- | --- | --- | ---: |",
    ]
    for spec in datasets:
        framework_rows = aggregated.get(spec.key, {})
        winner, advantage = determine_best_framework(spec.metric_name, framework_rows, score_field=score_field)
        lines.append(f"| `{spec.key}` | `{spec.metric_name}` | `{winner}` | {fmt_score(advantage)} |")

    lines.extend(
        [
            "",
            "## Framework Metrics",
            "",
            f"| Dataset | Framework | Primary Metric | {score_label} | Reproducibility Std | Runtime Mean (s) | Runtime Std | Valid Output | Input Token Cost | Output Token Cost | Token Total | LLM Call Count |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for spec in datasets:
        for framework in FRAMEWORKS:
            row = (aggregated.get(spec.key) or {}).get(framework) or {}
            lines.append(
                "| `{dataset}` | `{framework}` | `{metric}` | {score} | {score_std} | {runtime} | {runtime_std} | `{valid}` | {input_tokens} | {output_tokens} | {total_tokens} | {calls} |".format(
                    dataset=spec.key,
                    framework=framework,
                    metric=row.get("metric_name", spec.metric_name),
                    score=fmt_score(row.get(score_field)),
                    score_std=fmt_score(row.get("reproducibility_std")),
                    runtime=fmt_score(row.get("runtime_mean_sec")),
                    runtime_std=fmt_score(row.get("runtime_std")),
                    valid=row.get("valid_output_rate", "-"),
                    input_tokens=fmt_score(row.get("input_token_cost")),
                    output_tokens=fmt_score(row.get("output_token_cost")),
                    total_tokens=fmt_score(row.get("token_total")),
                    calls=fmt_score(row.get("llm_call_count")),
                )
            )

    for spec in datasets:
        lines.extend(
            [
                "",
                f"## `{spec.key}`",
                "",
                f"- Description: `{spec.description}`",
                f"- Primary metric: `{spec.metric_name}`",
            ]
        )
        framework_rows = aggregated.get(spec.key, {})
        winner, advantage = determine_best_framework(spec.metric_name, framework_rows, score_field=score_field)
        lines.extend([f"- Better framework: `{winner}`", f"- Advantage: `{fmt_score(advantage)}`", ""])
        for framework in FRAMEWORKS:
            row = framework_rows.get(framework) or {}
            lines.extend(
                [
                    f"### {framework}",
                    "",
                    f"- {score_label}: `{fmt_score(row.get(score_field))}`",
                    f"- Score mean: `{fmt_score(row.get('score_mean'))}`",
                    f"- Reproducibility std: `{fmt_score(row.get('reproducibility_std'))}`",
                    f"- Runtime mean (s): `{fmt_score(row.get('runtime_mean_sec'))}`",
                    f"- Runtime std: `{fmt_score(row.get('runtime_std'))}`",
                    f"- Valid output rate: `{row.get('valid_output_rate', '-')}`",
                    f"- Success rate: `{row.get('success_rate', '-')}`",
                    f"- Input token cost: `{fmt_score(row.get('input_token_cost'))}`",
                    f"- Output token cost: `{fmt_score(row.get('output_token_cost'))}`",
                    f"- Token total: `{fmt_score(row.get('token_total'))}`",
                    f"- LLM call count: `{fmt_score(row.get('llm_call_count'))}`",
                    f"- Representative prediction file: `{row.get('prediction_file', '-')}`",
                    f"- Representative run dir: `{row.get('run_dir', '-')}`",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def run_single_job(
    framework: str,
    spec: DatasetSpec,
    repeat_index: int,
    args: argparse.Namespace,
    raw_log_path: Path,
    raw_log_lock: threading.Lock,
    framework_lock: threading.Lock,
) -> dict[str, Any]:
    usage_log = TMP_DIR / "usage_logs" / f"{framework}-{spec.key}-r{repeat_index + 1}.jsonl"
    with framework_lock:
        if framework == "main":
            result = run_main_dataset(spec, args.env_file, args.main_timeout, usage_log)
        elif framework == "aide":
            result = run_aide_dataset(spec, args.env_file, args.aide_timeout, args.aide_steps, usage_log)
        elif framework == "lightautods_lama":
            result = run_lads_dataset(spec, args.env_file, args.lads_timeout, usage_log)
        elif framework == "autokaggle":
            result = run_autokaggle_dataset(spec, args.env_file, args.autokaggle_timeout, args.valid_size, usage_log)
        else:
            raise KeyError(framework)

    result["dataset"] = spec.key
    result["repeat"] = repeat_index + 1
    result["generated_at"] = utc_now()
    with raw_log_lock:
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def build_output_paths(output_stem: str) -> tuple[Path, Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if output_stem:
        stem = output_stem
    else:
        stem = f"framework-matrix-extended-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    md_path = REPORT_DIR / f"{stem}.md"
    json_path = REPORT_DIR / f"{stem}.json"
    raw_jsonl = TMP_DIR / f"{stem}-runs.jsonl"
    return md_path, json_path, raw_jsonl


def main() -> int:
    args = parse_args()
    dataset_specs = [DATASETS[key] for key in args.datasets] if args.datasets else [DATASETS[spec.key] for spec in ALL_VALIDATION_DATASETS]
    md_path, json_path, raw_jsonl = build_output_paths(args.output_stem)
    print(f"[FRAMEWORK-MATRIX-EXTENDED] writing to {md_path}", flush=True)

    framework_locks = {framework: threading.Lock() for framework in FRAMEWORKS}
    raw_log_lock = threading.Lock()
    futures = []
    results: list[dict[str, Any]] = []
    completed_jobs: set[tuple[str, str, int]] = set()

    if args.resume and raw_jsonl.exists():
        results = load_jsonl(raw_jsonl)
        for row in results:
            dataset = str(row.get("dataset") or "").strip()
            framework = str(row.get("framework") or "").strip()
            repeat = int(row.get("repeat") or 0)
            if dataset and framework and repeat > 0:
                completed_jobs.add((dataset, framework, repeat))
        print(
            f"[FRAMEWORK-MATRIX-EXTENDED] resume enabled: loaded {len(results)} prior run rows from {raw_jsonl}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        for repeat_index in range(args.repeats):
            for spec in dataset_specs:
                for framework in args.frameworks:
                    job_key = (spec.key, framework, repeat_index + 1)
                    if job_key in completed_jobs:
                        print(
                            f"[SKIP] {framework} {spec.key} r{repeat_index + 1} already present in {raw_jsonl.name}",
                            flush=True,
                        )
                        continue
                    futures.append(
                        executor.submit(
                            run_single_job,
                            framework,
                            spec,
                            repeat_index,
                            args,
                            raw_jsonl,
                            raw_log_lock,
                            framework_locks[framework],
                        )
                    )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                "[DONE] {framework} {dataset} r{repeat} status={status} metric={metric} runtime={runtime}s tokens={tokens} calls={calls}".format(
                    framework=result["framework"],
                    dataset=result["dataset"],
                    repeat=result["repeat"],
                    status=result.get("status"),
                    metric=fmt_score(result.get("metric_value")),
                    runtime=fmt_score(result.get("runtime_sec")),
                    tokens=fmt_score(result.get("token_total")),
                    calls=fmt_score(result.get("llm_call_count")),
                ),
                flush=True,
            )

    aggregated: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in dataset_specs:
        aggregated[spec.key] = {}
        for framework in FRAMEWORKS:
            runs = [
                row
                for row in results
                if row.get("dataset") == spec.key and row.get("framework") == framework
            ]
            aggregated[spec.key][framework] = aggregate_runs(spec.key, framework, sorted(runs, key=lambda item: item["repeat"]))

    payload = {
        "generated_at": utc_now(),
        "datasets": [spec.key for spec in dataset_specs],
        "frameworks": list(FRAMEWORKS),
        "repeats": args.repeats,
        "aggregated": aggregated,
        "raw_results_count": len(results),
        "raw_results_path": str(raw_jsonl),
    }
    save_json(json_path, payload)
    md_text = render_markdown(
        "# Framework Matrix Extended",
        dataset_specs,
        aggregated,
        payload["generated_at"],
    )
    md_path.write_text(md_text, encoding="utf-8")
    print(f"[FRAMEWORK-MATRIX-EXTENDED] markdown={md_path}", flush=True)
    print(f"[FRAMEWORK-MATRIX-EXTENDED] json={json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
