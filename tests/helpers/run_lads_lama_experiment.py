#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import dotenv_values


JSON_REGEX = r"```json\s*(.+?)```"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LightAutoDS LightAutoML-route experiment and emit JSON.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--task-family", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--env-file", default="")
    parser.add_argument("--use-parser", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_workspace(repo_root: Path, data_dir: Path, run_root: Path) -> pd.DataFrame:
    datasets_dir = repo_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    for child in datasets_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        shutil.copy2(data_dir / name, datasets_dir / name)
    for optional in ("overview.txt", "data_description.txt"):
        src = data_dir / optional
        if src.exists():
            shutil.copy2(src, datasets_dir / optional)
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    return pd.read_csv(datasets_dir / "train.csv")


def _default_config(task_family: str, target: str) -> dict[str, str]:
    if task_family == "regression":
        return {"task_type": "reg", "target": target, "task_metric": "r2"}
    return {"task_type": "binary", "target": target, "task_metric": "auc"}


def _parse_config_from_text(text: str, task_family: str, target: str) -> dict[str, str]:
    try:
        matches = re.findall(JSON_REGEX, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            payload = json.loads(matches[-1].strip())
        else:
            payload = json.loads(text.strip())
        config = _default_config(task_family, target)
        normalized: dict[str, str] = {}
        for key, value in payload.items():
            if key not in {"task_type", "target", "task_metric"}:
                continue
            if key == "target" and isinstance(value, list):
                normalized[key] = ",".join(str(item) for item in value)
            else:
                normalized[key] = str(value)
        config.update(normalized)
        if config.get("task_type") == "reg" and config.get("task_metric") == "r2-score":
            config["task_metric"] = "r2"
        if config.get("task_type") == "reg" and config.get("task_metric") == "rmse":
            config["task_metric"] = "mse"
        return config
    except Exception:
        return _default_config(task_family, target)


def _extract_metric(stdout: str) -> float | None:
    match = re.search(r"(?im)^\s*METRIC_VALUE:\s*(-?\d+(?:\.\d+)?)\s*$", stdout)
    if not match:
        return None
    return float(match.group(1))


def _load_env_file(env_file: str) -> None:
    if not env_file:
        return
    values = dotenv_values(env_file)
    for key, value in values.items():
        if value is None:
            continue
        os.environ.setdefault(key, str(value))
    if not os.environ.get("OPENAI_API_KEY"):
        api_key = values.get("LLM_MINIMAX_API_KEY")
        if api_key:
            os.environ["OPENAI_API_KEY"] = str(api_key)
    if not os.environ.get("OPENAI_BASE_URL"):
        base_url = values.get("LLM_MINIMAX_BASE_URL")
        if base_url:
            os.environ["OPENAI_BASE_URL"] = str(base_url)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output_json).resolve()
    run_root = output_path.with_suffix("")
    started = time.time()

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    _load_env_file(args.env_file)

    try:
        train_df = _prepare_workspace(repo_root, data_dir, run_root)
        response_text = ""
        lama_config = _default_config(args.task_family, args.target)

        if args.use_parser:
            from utils.config.loader import load_config
            from utils.llm_factory import create_llm
            from graph.llm_nodes import lightautoml_generator

            config = load_config()
            llm = create_llm("lightautoml_config_generator", config)
            state = {
                "task": (
                    f"Use LightAutoML to {args.task}. "
                    f"Target column(s): {args.target}. "
                    f"Dataset description: {args.description}. "
                    f"Preferred benchmark metric: {args.metric_name}."
                ),
                "df": train_df,
                "df_name": "train.csv",
            }
            response = lightautoml_generator(state, llm)
            response_text = getattr(response["messages"], "content", "")
            lama_config = _parse_config_from_text(response_text, args.task_family, args.target)

        model_dir = run_root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(repo_root / "graph" / "lightautoml_template.py"),
            "--df_name",
            "train.csv",
            "--task_type",
            lama_config["task_type"],
            "--task_metric",
            lama_config["task_metric"],
            "--target",
            lama_config["target"],
            "--external_metric",
            args.metric_name,
            "--output_dir",
            str(model_dir),
        ]
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=600,
        )

        metric_value = _extract_metric(proc.stdout or "")
        payload = {
            "status": "PASS" if proc.returncode == 0 and metric_value is not None else "FAIL",
            "runtime_sec": round(time.time() - started, 4),
            "metric_name": args.metric_name,
            "metric_value": metric_value,
            "internal_config": lama_config,
            "parser_response": response_text,
            "route": "LAMA",
            "run_dir": str(run_root),
            "prediction_file": str(model_dir / "submission.csv"),
            "prediction_exists": (model_dir / "submission.csv").exists(),
            "validation_predictions_file": str(model_dir / "validation_predictions.csv"),
            "validation_predictions_exists": (model_dir / "validation_predictions.csv").exists(),
            "metrics_json": str(model_dir / "metrics.json"),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "error": "" if proc.returncode == 0 else f"template_returncode={proc.returncode}",
        }
        _safe_write_json(output_path, payload)
        print(f"LADS_LAMA_RESULT_JSON::{output_path}")
        return 0 if payload["status"] == "PASS" else 1
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "runtime_sec": round(time.time() - started, 4),
            "metric_name": args.metric_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "route": "LAMA",
            "run_dir": str(run_root),
        }
        _safe_write_json(output_path, payload)
        print(f"LADS_LAMA_RESULT_JSON::{output_path}")
        print(payload["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
