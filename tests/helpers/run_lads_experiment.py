#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LightAutoDS experiment and emit JSON.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--task-family", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_text_if_exists(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:limit]


def _infer_direction(metric_name: str) -> str:
    metric = metric_name.lower().strip()
    if metric.startswith("neg_"):
        return "maximize"
    if any(token in metric for token in ("error", "loss", "rmse", "mae", "mse")):
        return "minimize"
    return "maximize"


def _extract_metric_candidates(text: str, metric_name: str) -> list[float]:
    candidates: list[float] = []
    patterns = [r"(?im)^\s*METRIC_VALUE:\s*(-?\d+(?:\.\d+)?)\s*$"]

    metric = metric_name.lower().strip()
    if metric == "accuracy":
        patterns.extend([r"(?im)^[^\n\r]*accuracy[^\n\r:]*[:=]\s*(-?\d+(?:\.\d+)?)\s*$"])
    elif metric == "roc_auc":
        patterns.extend([r"(?im)^[^\n\r]*roc[-\s_]*auc[^\n\r:]*[:=]\s*(-?\d+(?:\.\d+)?)\s*$"])
    elif metric == "f1_macro":
        patterns.extend(
            [
                r"(?im)^[^\n\r]*f1[^\n\r]*macro[^\n\r:]*[:=]\s*(-?\d+(?:\.\d+)?)\s*$",
                r"(?im)^[^\n\r]*macro[^\n\r]*f1[^\n\r:]*[:=]\s*(-?\d+(?:\.\d+)?)\s*$",
            ]
        )
    elif metric == "roc_auc_macro_ovr":
        patterns.extend(
            [
                r"(?im)^[^\n\r]*roc[-\s_]*auc[^\n\r:]*[:=]\s*(-?\d+(?:\.\d+)?)\s*$",
                r"(?im)^[^\n\r]*macro[^\n\r]*auc[^\n\r:]*[:=]\s*(-?\d+(?:\.\d+)?)\s*$",
            ]
        )
    elif metric == "neg_root_mean_squared_error":
        patterns.extend(
            [
                r"(?im)^[^\n\r]*neg[_\s-]*root[_\s-]*mean[_\s-]*squared[_\s-]*error[^\n\r:]*[:=]\s*(-?\d+(?:\.\d+)?)\s*$",
                r"(?im)^[^\n\r]*rmse[^\n\r:]*[:=]\s*(-?\d+(?:\.\d+)?)\s*$",
            ]
        )

    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            try:
                candidates.append(float(match))
            except Exception:
                pass

    if metric == "neg_root_mean_squared_error":
        # If we only found positive RMSE values, convert them to neg RMSE to align with the main metric.
        converted: list[float] = []
        for value in candidates:
            converted.append(-value if value > 0 else value)
        candidates = converted

    return candidates


def _choose_best_metric(values: list[float], metric_name: str) -> float | None:
    if not values:
        return None
    direction = _infer_direction(metric_name)
    return max(values) if direction == "maximize" else min(values)


def _build_task_prompt(task: str, target: str, metric_name: str, task_family: str, description: str, overview_text: str) -> str:
    target_cols = ", ".join(part.strip() for part in target.split(","))
    metric_instructions = {
        "accuracy": (
            "Evaluate on a validation split from train.csv using accuracy with class-label predictions."
        ),
        "roc_auc": (
            "Evaluate on a validation split from train.csv using ROC AUC with predicted probabilities."
        ),
        "f1_macro": (
            "Evaluate on a validation split from train.csv using macro-averaged F1 with class-label predictions."
        ),
        "roc_auc_macro_ovr": (
            "Evaluate on a validation split from train.csv using macro-averaged one-vs-rest ROC AUC. "
            "For multilabel targets, use predicted probabilities for every target column."
        ),
        "neg_root_mean_squared_error": (
            "Evaluate on a validation split from train.csv using negative root mean squared error."
        ),
    }
    metric_text = metric_instructions.get(metric_name, f"Evaluate on a validation split from train.csv using `{metric_name}`.")
    prompt = (
        "CUSTOM_CODE_ONLY\n"
        f"Build a tabular machine learning pipeline to {task}.\n"
        f"Task type: {description}.\n"
        f"Target column(s): {target_cols}.\n"
        "Use custom generated Python code and local ML libraries only.\n"
        "Use `datasets/train.csv` for development, `datasets/test.csv` for final inference, "
        "and match the format of `datasets/sample_submission.csv` exactly.\n"
        f"{metric_text}\n"
        "Print exactly one line in the format `METRIC_VALUE: <number>` for the best validation metric you achieved.\n"
        "After validation, train the final pipeline and save the final test predictions to a CSV file.\n"
        "Prefer robust libraries available locally such as pandas, scikit-learn, xgboost, lightgbm, and catboost. "
        "Avoid asking for user input.\n"
    )
    if task_family == "multilabel_classification":
        prompt += (
            "This is a multilabel task. Predict every target column independently or with a multilabel-capable estimator, "
            "and make sure the submission contains one prediction column per target.\n"
        )
    if overview_text:
        prompt += f"\nCompetition overview excerpt:\n{overview_text}\n"
    return prompt


def _prepare_run_workspace(repo_root: Path, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets_dir = repo_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for child in datasets_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    for extra in ("code", "models", "output"):
        extra_path = repo_root / extra
        if extra_path.exists():
            shutil.rmtree(extra_path)

    for pattern in ("*_predictions.csv", "submission.csv"):
        for file_path in repo_root.glob(pattern):
            if file_path.is_file():
                file_path.unlink()

    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        shutil.copy2(data_dir / name, datasets_dir / name)
    for optional in ("overview.txt", "data_description.txt"):
        src = data_dir / optional
        if src.exists():
            shutil.copy2(src, datasets_dir / optional)

    train_df = pd.read_csv(datasets_dir / "train.csv")
    test_df = pd.read_csv(datasets_dir / "test.csv")
    return train_df, test_df


def _collect_prediction_file(repo_root: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("*_predictions.csv", "submission.csv", "code/*.csv"):
        candidates.extend(path for path in repo_root.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _should_stop_early(node_name: str, content: str, repo_root: Path, metric_trace: list[float]) -> bool:
    if not metric_trace:
        return False
    prediction_file = _collect_prediction_file(repo_root)
    if prediction_file is None:
        return False
    if node_name == "check_train_test_inference" and "VALID" in content:
        return True
    if node_name == "execute_train_test" and "Результаты выполнения кода для тестирования" in content:
        return True
    return False


def _append_state(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    for key, value in update.items():
        if key == "messages":
            existing = list(merged.get("messages") or [])
            if isinstance(value, list):
                existing.extend(value)
            else:
                existing.append(value)
            merged["messages"] = existing
        else:
            merged[key] = value
    return merged


def _latest_message_content(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    if not messages:
        return ""
    return getattr(messages[-1], "content", "") or ""


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output_json).resolve()
    run_root = output_path.with_suffix("")
    started = time.time()

    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    try:
        from graph.code_executor_node import execute_code, execute_train_test
        from graph.llm_nodes import (
            code_generation_agent,
            input_node,
            rephraser_agent,
            train_inference_split,
            check_train_test_inference,
        )
        from utils.config.loader import load_config
        from utils.llm_factory import create_llm

        train_df, test_df = _prepare_run_workspace(repo_root, data_dir)
        overview = _read_text_if_exists(data_dir / "overview.txt")
        task_prompt = _build_task_prompt(
            task=args.task,
            target=args.target,
            metric_name=args.metric_name,
            task_family=args.task_family,
            description=args.description,
            overview_text=overview,
        )

        config = load_config()
        state_input = {
            "messages": [HumanMessage(content=task_prompt)],
            "code_generation_config": config.general.code_generation_config,
            "df": train_df,
            "df_name": "train.csv",
            "test_df": test_df,
            "test_df_name": "test.csv",
        }

        metric_trace: list[float] = []
        nodes_visited: list[str] = []
        state = input_node(state_input)
        state["current_node"] = "input_node"
        stopped_early = False
        nodes_visited.append("input_node")

        llm_plan = create_llm("rephraser_agent", config)
        state = _append_state(state, rephraser_agent(state, llm_plan))
        state["current_node"] = "rephraser_agent"
        nodes_visited.append("rephraser_agent")

        llm_codegen = create_llm("code_generation_agent", config)
        code_attempts = 3
        for attempt in range(code_attempts):
            state = _append_state(state, code_generation_agent(state, llm_codegen))
            state["current_node"] = "code_generation_agent"
            nodes_visited.append("code_generation_agent")

            state = _append_state(state, execute_code(state))
            state["current_node"] = "code_executor"
            nodes_visited.append("code_executor")

            content = _latest_message_content(state)
            metric_trace.extend(_extract_metric_candidates(content, args.metric_name))
            if metric_trace and "возникла ошибка" not in content.lower() and "traceback" not in content.lower():
                break
            if attempt < code_attempts - 1:
                state = _append_state(
                    state,
                    {
                        "messages": [
                            HumanMessage(
                                content=(
                                    "The previous code execution was not good enough. "
                                    "Rewrite the full solution, fix the execution issues, and make sure you print "
                                    "`METRIC_VALUE: <number>` exactly once."
                                )
                            )
                        ]
                    },
                )

        llm_split = create_llm("train_inference_splitter", config)
        state = _append_state(state, train_inference_split(state, llm_split))
        state["current_node"] = "train_inference_splitter"
        nodes_visited.append("train_inference_splitter")

        llm_checker = create_llm("check_train_test_inference", config)
        split_attempts = 2
        for _ in range(split_attempts):
            state = _append_state(state, execute_train_test(state))
            state["current_node"] = "execute_train_test"
            nodes_visited.append("execute_train_test")

            content = _latest_message_content(state)
            if _should_stop_early("execute_train_test", content, repo_root, metric_trace):
                stopped_early = True
                break

            state = _append_state(state, check_train_test_inference(state, llm_checker))
            state["current_node"] = "check_train_test_inference"
            nodes_visited.append("check_train_test_inference")
            content = _latest_message_content(state)
            if _should_stop_early("check_train_test_inference", content, repo_root, metric_trace):
                stopped_early = True
                break

        final_values = state

        best_metric = _choose_best_metric(metric_trace, args.metric_name)
        route = "CODEGEN"

        run_root.mkdir(parents=True, exist_ok=True)
        prediction_src = _collect_prediction_file(repo_root)
        prediction_dst = None
        if prediction_src:
            prediction_dst = run_root / prediction_src.name
            shutil.copy2(prediction_src, prediction_dst)

        for folder_name in ("code", "models", "output"):
            src = repo_root / folder_name
            if src.exists():
                dst = run_root / folder_name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        final_message = ""
        final_messages = final_values.get("messages")
        if final_messages:
            final_message = getattr(final_messages[-1], "content", "")

        has_error_signal = (
            "traceback" in final_message.lower()
            or "возникла ошибка" in final_message.lower()
            or "an error occurred" in final_message.lower()
        )
        status = "PASS" if best_metric is not None and not has_error_signal else "FAIL"

        payload = {
            "status": status,
            "runtime_sec": round(time.time() - started, 4),
            "metric_name": args.metric_name,
            "metric_value": best_metric,
            "metric_trace": metric_trace,
            "route": route,
            "nodes_visited": nodes_visited,
            "prediction_file": str(prediction_dst) if prediction_dst else "",
            "prediction_exists": bool(prediction_dst and prediction_dst.exists()),
            "run_dir": str(run_root),
            "final_message": final_message,
            "feedback_count": len(final_values.get("feedback") or []),
            "human_understanding_count": len(final_values.get("human_understanding") or []),
            "generated_code_length": len(final_values.get("generated_code") or ""),
            "stopped_early": stopped_early,
        }
        _safe_write_json(output_path, payload)
        print(f"LADS_RESULT_JSON::{output_path}")
        return 0
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "runtime_sec": round(time.time() - started, 4),
            "metric_name": args.metric_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "run_dir": str(run_root),
        }
        _safe_write_json(output_path, payload)
        print(f"LADS_RESULT_JSON::{output_path}")
        print(payload["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
