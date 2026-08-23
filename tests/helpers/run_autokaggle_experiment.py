#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import dotenv_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AutoKaggle validation-backed experiment and emit JSON.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--autokaggle-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--task-family", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--env-file", default="")
    parser.add_argument("--model", default="MiniMax-M2.7")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--timeout-sec", type=int, default=2400)
    return parser.parse_args()


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_env_file(env_file: str) -> dict[str, str]:
    if not env_file:
        return {}
    values = dotenv_values(env_file)
    loaded: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        loaded[key] = str(value)
        os.environ.setdefault(key, str(value))
    return loaded


def _resolve_model_name(cli_model: str, env_values: dict[str, str]) -> str:
    configured = env_values.get("AGENT_MODEL_BUILDING_MODEL", "").strip()
    candidate = configured or cli_model
    return candidate.split("/", 1)[1] if "/" in candidate else candidate


def _target_columns(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _maybe_stratify(df: pd.DataFrame, target_cols: list[str], task_family: str):
    if "classification" not in task_family:
        return None
    try:
        if len(target_cols) == 1:
            series = df[target_cols[0]]
            counts = series.value_counts(dropna=False)
            if len(counts) > 1 and int(counts.min()) >= 2:
                return series
            return None
        signature = df[target_cols].astype(str).agg("__".join, axis=1)
        counts = signature.value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            return signature
    except Exception:
        return None
    return None


def _split_train_valid(
    train_df: pd.DataFrame,
    target_cols: list[str],
    task_family: str,
    valid_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    stratify = _maybe_stratify(train_df, target_cols, task_family)
    train_part, valid_part = train_test_split(
        train_df,
        test_size=valid_size,
        random_state=42,
        shuffle=True,
        stratify=stratify,
    )
    return train_part.reset_index(drop=True), valid_part.reset_index(drop=True)


def _build_competition_inputs(
    source_dir: Path,
    competition_dir: Path,
    target_cols: list[str],
    task: str,
    metric_name: str,
    description: str,
    task_family: str,
    valid_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(source_dir / "train.csv")
    original_sample = pd.read_csv(source_dir / "sample_submission.csv")
    id_col = original_sample.columns[0]

    train_part, valid_part = _split_train_valid(train_df, target_cols, task_family, valid_size)
    valid_features = valid_part.drop(columns=target_cols)

    competition_dir.mkdir(parents=True, exist_ok=True)
    train_part.to_csv(competition_dir / "train.csv", index=False)
    valid_features.to_csv(competition_dir / "test.csv", index=False)

    sample_submission = pd.DataFrame({id_col: valid_part[id_col].to_list()})
    for col in target_cols:
        sample_submission[col] = valid_part[col].to_list()
    sample_submission.to_csv(competition_dir / "sample_submission.csv", index=False)

    overview_parts: list[str] = []
    for optional in ("overview.txt", "data_description.txt"):
        src = source_dir / optional
        if src.exists():
            overview_parts.append(src.read_text(encoding="utf-8"))
    overview_parts.append(
        "\n\n# Validation Benchmark Note\n"
        f"- Task: {task}\n"
        f"- Dataset type: {description}\n"
        f"- Primary evaluation metric: {metric_name}\n"
        "- `train.csv` is the supervised training split.\n"
        "- `test.csv` is a held-out validation feature split for this benchmark harness.\n"
        "- `sample_submission.csv` defines the exact row order and column schema required for scoring.\n"
        "- You must write `submission.csv` in the competition root.\n"
    )
    if metric_name in {"roc_auc", "roc_auc_macro_ovr"}:
        overview_parts.append(
            "- Prefer probability outputs in `[0, 1]` for the target column(s). "
            "If that is not possible, numeric confidence scores are acceptable."
        )
    elif metric_name in {"accuracy", "f1_macro"}:
        overview_parts.append("- Output final class labels in the target column(s).")
    else:
        overview_parts.append("- Output predictions on the original target scale.")
    (competition_dir / "overview.txt").write_text("\n".join(overview_parts), encoding="utf-8")
    return train_part, valid_part, sample_submission


def _configure_api_files(autokaggle_root: Path, env_values: dict[str, str], model_name: str) -> None:
    api_key = env_values.get("LLM_MINIMAX_API_KEY") or os.environ.get("LLM_MINIMAX_API_KEY", "")
    base_url = env_values.get("LLM_MINIMAX_BASE_URL") or os.environ.get("LLM_MINIMAX_BASE_URL", "")
    if not api_key:
        raise RuntimeError("LLM_MINIMAX_API_KEY is missing; cannot run AutoKaggle with MiniMax.")
    (autokaggle_root / "api_key.txt").write_text(f"{api_key}\n{base_url}\n", encoding="utf-8")
    os.environ["AUTOKAGGLE_DEFAULT_MODEL"] = model_name
    os.environ.setdefault("AUTOKAGGLE_MAX_COMPLETION_TOKENS", "1024")
    os.environ.setdefault("AUTOKAGGLE_MAX_ITERATIONS", "2")
    os.environ.setdefault("AUTOKAGGLE_RETRY_DELAY_SEC", "5")
    os.environ.setdefault("AUTOKAGGLE_API_TIMEOUT_SEC", "90")
    os.environ.setdefault("AUTOKAGGLE_FAST_COMPAT_READER", "1")
    os.environ.setdefault("AUTOKAGGLE_FAST_COMPAT_PLANNER", "1")
    os.environ.setdefault("AUTOKAGGLE_FAST_COMPAT_REVIEWER", "1")
    os.environ.setdefault("AUTOKAGGLE_FAST_COMPAT_SUMMARIZER", "1")
    os.environ.setdefault("AUTOKAGGLE_FAST_COMPAT_DEVELOPER", "1")
    os.environ["AUTOKAGGLE_BENCHMARK_METRIC"] = os.environ.get("AUTOKAGGLE_BENCHMARK_METRIC", "")
    for role in ("reader", "planner", "developer", "reviewer", "summarizer"):
        os.environ.setdefault(f"AUTOKAGGLE_{role.upper()}_MODEL", model_name)


def _validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if list(submission.columns) != list(sample_submission.columns):
        issues.append("column_mismatch")
    if len(submission) != len(sample_submission):
        issues.append("row_count_mismatch")
    if submission.isna().any().any():
        issues.append("contains_nan")
    return (not issues), issues


def _coerce_binary_scores(series: pd.Series, y_true: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).to_numpy()
    values = [value for value in pd.Series(y_true).dropna().unique().tolist()]
    if len(values) != 2:
        raise ValueError("Binary ROC AUC requires exactly two target classes.")
    positive = values[-1]
    return series.map(lambda value: 1.0 if value == positive else 0.0).astype(float).to_numpy()


def _score_predictions(
    metric_name: str,
    task_family: str,
    y_true_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    target_cols: list[str],
) -> float:
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

    if metric_name == "accuracy":
        return float(accuracy_score(y_true_df[target_cols[0]], submission_df[target_cols[0]]))
    if metric_name == "f1_macro":
        return float(
            f1_score(
                y_true_df[target_cols[0]],
                submission_df[target_cols[0]],
                average="macro",
                zero_division=0,
            )
        )
    if metric_name == "roc_auc":
        y_true = y_true_df[target_cols[0]]
        scores = _coerce_binary_scores(submission_df[target_cols[0]], y_true)
        return float(roc_auc_score(y_true, scores))
    if metric_name == "neg_root_mean_squared_error":
        y_true = y_true_df[target_cols[0]].astype(float).to_numpy()
        y_pred = submission_df[target_cols[0]].astype(float).to_numpy()
        return -float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if metric_name == "roc_auc_macro_ovr":
        y_true = y_true_df[target_cols].astype(float).to_numpy()
        y_score = submission_df[target_cols].astype(float).to_numpy()
        try:
            return float(roc_auc_score(y_true, y_score, average="macro"))
        except Exception:
            per_target: list[float] = []
            for index, col in enumerate(target_cols):
                try:
                    per_target.append(
                        float(roc_auc_score(y_true_df[col].astype(float), submission_df[col].astype(float)))
                    )
                except Exception:
                    continue
            if not per_target:
                raise
            return float(sum(per_target) / len(per_target))
    raise ValueError(f"Unsupported benchmark metric: {metric_name} for {task_family}")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    autokaggle_root = Path(args.autokaggle_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_json = Path(args.output_json).resolve()
    run_root = output_json.with_suffix("")
    competition_name = run_root.name
    competition_dir = autokaggle_root / "multi_agents" / "competition" / competition_name
    target_cols = _target_columns(args.target)
    started = time.time()

    try:
        env_values = _load_env_file(args.env_file)
        model_name = _resolve_model_name(args.model, env_values)
        _configure_api_files(autokaggle_root, env_values, model_name)
        os.environ["AUTOKAGGLE_BENCHMARK_METRIC"] = args.metric_name
        os.environ["AUTOKAGGLE_TASK_FAMILY"] = args.task_family
        os.environ["AUTOKAGGLE_TARGET_COLUMNS"] = ",".join(target_cols)
        if competition_dir.exists():
            shutil.rmtree(competition_dir)
        _, valid_part, sample_submission = _build_competition_inputs(
            data_dir,
            competition_dir,
            target_cols,
            args.task,
            args.metric_name,
            args.description,
            args.task_family,
            args.valid_size,
        )
        python_bin = autokaggle_root.parent / ".venv" / "bin" / "python"
        if not python_bin.exists():
            raise RuntimeError(f"AutoKaggle venv not found: {python_bin}")
        cmd = [
            str(python_bin),
            str(autokaggle_root / "framework.py"),
            "--competition",
            competition_name,
            "--model",
            model_name.replace("-", "_"),
        ]
        proc = subprocess.run(
            cmd,
            cwd=autokaggle_root,
            text=True,
            capture_output=True,
            timeout=args.timeout_sec,
            env=os.environ.copy(),
        )
        submission_path = competition_dir / "submission.csv"
        submission_exists = submission_path.exists()
        submission_df = pd.read_csv(submission_path) if submission_exists else pd.DataFrame()
        valid_output, valid_issues = _validate_submission(submission_df, sample_submission) if submission_exists else (False, ["missing_submission"])
        metric_value = None
        if submission_exists and valid_output:
            metric_value = _score_predictions(
                args.metric_name,
                args.task_family,
                valid_part[target_cols],
                submission_df,
                target_cols,
            )
        payload = {
            "status": "PASS" if proc.returncode == 0 and metric_value is not None else "FAIL",
            "runtime_sec": round(time.time() - started, 4),
            "metric_name": args.metric_name,
            "metric_value": metric_value,
            "framework": "autokaggle",
            "route": "multi_agent_validation_harness",
            "model_name": model_name,
            "competition_name": competition_name,
            "competition_dir": str(competition_dir),
            "prediction_file": str(submission_path),
            "prediction_exists": submission_exists,
            "valid_output": valid_output,
            "valid_output_issues": valid_issues,
            "validation_rows": int(len(valid_part)),
            "target_columns": target_cols,
            "log_file": str(competition_dir / f"{competition_name}.log"),
            "research_report": str(competition_dir / "research_report.md"),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-3000:],
            "error": "" if proc.returncode == 0 and metric_value is not None else f"framework_returncode={proc.returncode}",
        }
        _safe_write_json(output_json, payload)
        print(f"AUTOKAGGLE_RESULT_JSON::{output_json}")
        return 0 if payload["status"] == "PASS" else 1
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "runtime_sec": round(time.time() - started, 4),
            "metric_name": args.metric_name,
            "framework": "autokaggle",
            "route": "multi_agent_validation_harness",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "competition_dir": str(competition_dir),
        }
        _safe_write_json(output_json, payload)
        print(f"AUTOKAGGLE_RESULT_JSON::{output_json}")
        print(payload["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
