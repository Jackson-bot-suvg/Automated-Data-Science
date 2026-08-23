"""Canonical task-contract helpers shared across workflow phases."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

CANONICAL_TASK_FAMILIES: tuple[str, ...] = (
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
)

_DEFAULT_CANDIDATE_METRICS: dict[str, list[str]] = {
    "binary_classification": ["roc_auc", "accuracy", "f1"],
    "multiclass_classification": ["f1_macro", "accuracy", "balanced_accuracy"],
    "regression": [
        "neg_root_mean_squared_error",
        "neg_mean_absolute_error",
        "r2",
    ],
    "multilabel_classification": ["roc_auc_macro_ovr", "f1_micro", "f1_macro"],
}

_METRIC_ALIASES: dict[str, dict[str, str]] = {
    "binary_classification": {
        "auc": "roc_auc",
        "roc_auc_score": "roc_auc",
        "f1_binary": "f1",
    },
    "multiclass_classification": {
        "f1": "f1_macro",
        "f1_weighted_macro": "f1_weighted",
        "roc_auc": "roc_auc_ovr_macro",
        "auc": "roc_auc_ovr_macro",
        "logloss": "neg_log_loss",
        "log_loss": "neg_log_loss",
    },
    "regression": {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    },
    "multilabel_classification": {
        "f1": "f1_micro",
        "auc": "roc_auc_macro_ovr",
        "roc_auc": "roc_auc_macro_ovr",
        "subset_acc": "subset_accuracy",
    },
}


def normalize_target_mode(
    raw_mode: str,
    *,
    target_count: int,
    task_family: str = "",
) -> str:
    """Return canonical target mode from mixed legacy and canonical values."""

    key = str(raw_mode or "").strip().lower().replace("-", "_")
    if key in {"single", "single_target"}:
        return "single_target"
    if key == "multi_output":
        return "multi_output"
    if key in {"multi", "multi_label", "multilabel"}:
        if target_count <= 1:
            return "single_target"
        if "regress" in task_family:
            return "multi_output"
        return "multi_label"
    if target_count <= 1:
        return "single_target"
    if "regress" in task_family:
        return "multi_output"
    return "multi_label"


def normalize_task_family(
    raw_family: str,
    *,
    target_mode: str = "",
    target_count: int = 0,
) -> str:
    """Map coarse or legacy task-family strings to canonical workflow labels."""

    family = str(raw_family or "").strip().lower().replace("-", "_").replace(" ", "_")
    mode = normalize_target_mode(target_mode, target_count=target_count, task_family=family)

    if family in CANONICAL_TASK_FAMILIES:
        if mode == "multi_label" and family != "multilabel_classification":
            return "multilabel_classification"
        return family
    if family in {"classification", "classif", "classifier"}:
        if mode == "multi_label" or target_count > 1:
            return "multilabel_classification"
        return "classification_unspecified"
    if "multilabel" in family or "multi_label" in family:
        return "multilabel_classification"
    if "multiclass" in family or "multi_class" in family:
        return "multiclass_classification"
    if "binary" in family:
        return "binary_classification"
    if "regress" in family:
        return "regression"
    return "unknown"


def infer_task_family_from_targets(
    train_df: pd.DataFrame,
    target_columns: list[str],
) -> str:
    """Infer canonical task family from observed target values."""

    if not target_columns:
        return "unknown"
    if len(target_columns) > 1:
        target_frame = train_df[target_columns]
        if all(_is_binary_like(target_frame[column]) for column in target_columns):
            return "multilabel_classification"
        if all(_looks_regression_like_numeric(target_frame[column]) for column in target_columns):
            return "regression"
        return "multilabel_classification"

    series = train_df[target_columns[0]]
    if _looks_regression_like_numeric(series):
        return "regression"
    if _is_binary_like(series):
        return "binary_classification"
    return "multiclass_classification"


def canonical_candidate_metrics(
    task_family: str,
    metrics: Iterable[str] | None = None,
) -> list[str]:
    """Normalize metric names for a canonical task family."""

    family = normalize_task_family(task_family, target_mode="", target_count=0)
    default_metrics = list(
        _DEFAULT_CANDIDATE_METRICS.get(family, _DEFAULT_CANDIDATE_METRICS["binary_classification"])
    )
    aliases = _METRIC_ALIASES.get(family, {})
    allowed = set(default_metrics) | set(aliases.values()) | {
        "accuracy",
        "balanced_accuracy",
        "f1",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "roc_auc",
        "roc_auc_ovr_macro",
        "roc_auc_macro_ovr",
        "average_precision",
        "neg_log_loss",
        "neg_root_mean_squared_error",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2",
        "subset_accuracy",
        "hamming_loss",
    }

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_metric in metrics or []:
        key = str(raw_metric or "").strip().lower().replace("-", "_")
        if not key:
            continue
        metric = aliases.get(key, key)
        if metric not in allowed or metric in seen:
            continue
        seen.add(metric)
        normalized.append(metric)

    if not normalized:
        return default_metrics

    primary = normalized[0]
    remainder = [metric for metric in normalized[1:] if metric != primary]
    return [primary] + remainder


def default_candidate_metrics(task_family: str) -> list[str]:
    family = normalize_task_family(task_family)
    return list(
        _DEFAULT_CANDIDATE_METRICS.get(family, _DEFAULT_CANDIDATE_METRICS["binary_classification"])
    )


def _is_binary_like(values: pd.Series) -> bool:
    series = pd.Series(values).dropna()
    if series.empty:
        return False
    unique_count = int(series.nunique())
    if unique_count > 2:
        return False
    normalized = {str(value).strip().lower() for value in series.unique().tolist()}
    if normalized.issubset({"0", "1", "true", "false", "yes", "no"}):
        return True
    return unique_count <= 2


def _looks_regression_like_numeric(values: pd.Series) -> bool:
    series = pd.Series(values).dropna()
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return False
    unique_count = int(series.nunique())
    sample_size = int(series.shape[0])
    if sample_size <= 0:
        return False
    has_fractional_values = ((series % 1) != 0).any()
    if has_fractional_values:
        return True
    unique_ratio = unique_count / sample_size
    if unique_count >= 20:
        return True
    if unique_count >= 5 and unique_ratio >= 0.5:
        return True
    return False


__all__ = [
    "CANONICAL_TASK_FAMILIES",
    "canonical_candidate_metrics",
    "default_candidate_metrics",
    "infer_task_family_from_targets",
    "normalize_target_mode",
    "normalize_task_family",
]
