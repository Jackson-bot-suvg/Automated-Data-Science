"""Metric computation tools — task-family aware.

Provides ``compute_metric_bundle`` with per-family dispatch:
    - binary_classification     → accuracy / f1 / precision / recall /
                                   roc_auc / average_precision
    - multiclass_classification → accuracy / f1_macro / f1_weighted /
                                   balanced_accuracy / log_loss /
                                   roc_auc_ovr_macro
    - regression                → mae / mse / rmse / r2 / mape /
                                   median_absolute_error
    - multilabel_classification → hamming_loss / f1_micro / f1_macro /
                                   f1_weighted / subset_accuracy /
                                   roc_auc_macro_ovr (if y_prob available)

Each helper is safe to call without sklearn when the computation degrades
naturally (e.g. accuracy via numpy). Where sklearn is required (AUC, log_loss),
failures degrade to ``None`` rather than raising.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Primitive helpers (kept for backward compatibility)
# ──────────────────────────────────────────────


def compute_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    if y_true.ndim > 1 and y_pred.ndim > 1 and y_true.shape == y_pred.shape:
        # Subset accuracy (all labels must match)
        return float(np.mean(np.all(y_true == y_pred, axis=1)))
    return float(np.mean(y_true == y_pred))


def compute_f1(y_true, y_pred, average: str = "binary") -> float:
    try:
        from sklearn.metrics import f1_score

        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except ImportError:
        pass

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def compute_roc_auc(y_true, y_prob) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_prob))


# ──────────────────────────────────────────────
# Task-family dispatchers
# ──────────────────────────────────────────────


def _safe(fn, *args, **kwargs):
    """Run fn and return float result or None on failure."""
    try:
        return round(float(fn(*args, **kwargs)), 4)
    except Exception as exc:  # noqa: BLE001
        logger.debug("metric %s failed: %s", getattr(fn, "__name__", fn), exc)
        return None


def _binary_bundle(y_true, y_pred, y_prob) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    bundle: dict[str, Any] = {
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "roc_auc": None,
        "average_precision": None,
    }
    if y_prob is not None:
        prob_vec = np.asarray(y_prob)
        if prob_vec.ndim == 2 and prob_vec.shape[1] == 2:
            prob_vec = prob_vec[:, 1]
        bundle["roc_auc"] = _safe(roc_auc_score, y_true, prob_vec)
        bundle["average_precision"] = _safe(
            average_precision_score, y_true, prob_vec
        )
    return bundle


def _multiclass_bundle(y_true, y_pred, y_prob) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    bundle: dict[str, Any] = {
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "f1_macro": _safe(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": _safe(
            f1_score, y_true, y_pred, average="weighted", zero_division=0
        ),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "log_loss": None,
        "roc_auc_ovr_macro": None,
    }
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] > 1:
            bundle["log_loss"] = _safe(log_loss, y_true, y_prob_arr)
            bundle["roc_auc_ovr_macro"] = _safe(
                roc_auc_score, y_true, y_prob_arr,
                multi_class="ovr", average="macro",
            )
    return bundle


def _regression_bundle(y_true, y_pred) -> dict[str, Any]:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    mse = _safe(mean_squared_error, y_true, y_pred)
    rmse = None
    if mse is not None and mse >= 0:
        rmse = round(float(np.sqrt(mse)), 4)

    return {
        "mae": _safe(mean_absolute_error, y_true, y_pred),
        "mse": mse,
        "rmse": rmse,
        "r2": _safe(r2_score, y_true, y_pred),
        "mape": _safe(mean_absolute_percentage_error, y_true, y_pred),
        "median_absolute_error": _safe(median_absolute_error, y_true, y_pred),
    }


def _multilabel_bundle(y_true, y_pred, y_prob, subtype: str = "") -> dict[str, Any]:
    from sklearn.metrics import (
        f1_score,
        hamming_loss,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)

    # When y is multiclass-multioutput, sklearn's f1_score/hamming_loss reject
    # the label type. Compute element-wise hamming + per-target macro-F1 mean
    # manually so the bundle stays informative.
    if subtype == "multiclass_multioutput":
        elementwise_mismatch = float(np.mean(y_true_arr != y_pred_arr))
        per_target_macro_f1: list[float] = []
        cols = min(y_true_arr.shape[1], y_pred_arr.shape[1])
        for j in range(cols):
            try:
                per_target_macro_f1.append(
                    float(
                        f1_score(
                            y_true_arr[:, j],
                            y_pred_arr[:, j],
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
            except Exception:  # noqa: BLE001
                pass
        return {
            "hamming_loss": round(elementwise_mismatch, 4),
            "subset_accuracy": _safe(
                lambda a, b: float(np.mean(np.all(a == b, axis=1))),
                y_true_arr, y_pred_arr,
            ),
            "per_target_macro_f1": (
                round(float(np.mean(per_target_macro_f1)), 4)
                if per_target_macro_f1 else None
            ),
            # The following are kept as None for downstream stability.
            "f1_micro": None,
            "f1_macro": None,
            "f1_weighted": None,
            "roc_auc_macro_ovr": None,
        }

    bundle: dict[str, Any] = {
        "hamming_loss": _safe(hamming_loss, y_true_arr, y_pred_arr),
        "f1_micro": _safe(
            f1_score, y_true_arr, y_pred_arr, average="micro", zero_division=0
        ),
        "f1_macro": _safe(
            f1_score, y_true_arr, y_pred_arr, average="macro", zero_division=0
        ),
        "f1_weighted": _safe(
            f1_score, y_true_arr, y_pred_arr, average="weighted", zero_division=0
        ),
        "subset_accuracy": _safe(
            lambda a, b: float(np.mean(np.all(a == b, axis=1))),
            y_true_arr, y_pred_arr,
        ),
        "roc_auc_macro_ovr": None,
    }
    if y_prob is not None:
        prob = _to_multilabel_prob_matrix(y_prob)
        if prob is not None and prob.shape == y_true_arr.shape:
            bundle["roc_auc_macro_ovr"] = _safe(
                roc_auc_score, y_true_arr, prob, average="macro"
            )
    return bundle


def _to_multilabel_prob_matrix(y_prob):
    if isinstance(y_prob, list):
        cols = []
        for arr in y_prob:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                cols.append(arr[:, 1])
            else:
                cols.append(arr.ravel())
        return np.column_stack(cols) if cols else None
    arr = np.asarray(y_prob)
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, 1) if arr.size else None


# ──────────────────────────────────────────────
# Public dispatcher
# ──────────────────────────────────────────────


def compute_metric_bundle(
    y_true,
    y_pred,
    y_prob=None,
    *,
    task_family: str = "binary_classification",
    target_mode: str = "single_target",
    multi_target_subtype: str = "",
) -> dict[str, Any]:
    """Compute a task-family aware metric bundle.

    Args:
        y_true: ground-truth labels (Series/ndarray/DataFrame for multilabel).
        y_pred: hard predictions (same shape family as y_true).
        y_prob: probability output (optional; required for AUC/log_loss).
        task_family: one of binary_classification / multiclass_classification /
            regression / multilabel_classification; coarse strings normalized.
        target_mode: single_target / multi_label / multi_output.

    Returns:
        Dict keyed by metric name; missing or failed metrics return None.
    """
    from app.tools.model_tools import normalize_task_family

    family = normalize_task_family(task_family)
    if target_mode == "multi_label":
        family = "multilabel_classification"

    if family == "regression":
        return _regression_bundle(y_true, y_pred)
    if family == "multilabel_classification":
        return _multilabel_bundle(y_true, y_pred, y_prob, subtype=multi_target_subtype)
    if family == "multiclass_classification":
        return _multiclass_bundle(y_true, y_pred, y_prob)
    # Default: binary (covers binary_classification & unknown single-label)
    return _binary_bundle(y_true, y_pred, y_prob)


__all__ = [
    "compute_accuracy",
    "compute_f1",
    "compute_roc_auc",
    "compute_metric_bundle",
]
