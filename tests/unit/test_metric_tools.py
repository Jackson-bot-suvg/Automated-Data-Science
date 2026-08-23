"""Unit tests for app/tools/metric_tools.py.

覆盖 4 个任务族：binary / multiclass / regression / multilabel。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.tools.metric_tools import (
    compute_accuracy,
    compute_f1,
    compute_metric_bundle,
    compute_roc_auc,
)


# ──────────────────────────────────────────────
# Primitive helpers (backward compatibility)
# ──────────────────────────────────────────────


def test_accuracy_basic():
    assert compute_accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)


def test_accuracy_empty():
    assert compute_accuracy([], []) == 0.0


def test_accuracy_subset_on_2d():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_pred = np.array([[1, 0], [1, 1], [1, 1]])
    # Only row 0 and row 2 fully match
    assert compute_accuracy(y_true, y_pred) == pytest.approx(2 / 3)


def test_f1_perfect():
    assert compute_f1([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_f1_no_positives_predicted():
    assert compute_f1([1, 0, 1], [0, 0, 0]) == 0.0


def test_roc_auc_basic():
    result = compute_roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert 0.9 <= result <= 1.0


# ──────────────────────────────────────────────
# compute_metric_bundle: binary_classification
# ──────────────────────────────────────────────


def test_bundle_binary_without_prob():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    b = compute_metric_bundle(y_true, y_pred, task_family="binary_classification")
    assert set(b) == {
        "accuracy", "f1", "precision", "recall", "roc_auc", "average_precision",
    }
    assert b["accuracy"] is not None
    assert b["roc_auc"] is None
    assert b["average_precision"] is None


def test_bundle_binary_with_prob():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.95])
    y_pred = (y_prob > 0.5).astype(int)
    b = compute_metric_bundle(
        y_true, y_pred, y_prob, task_family="binary_classification"
    )
    assert b["roc_auc"] is not None
    assert 0 <= b["roc_auc"] <= 1


def test_bundle_binary_accepts_2d_prob_with_two_columns():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
    y_pred = np.array([0, 1, 1, 0])
    b = compute_metric_bundle(
        y_true, y_pred, y_prob, task_family="binary_classification"
    )
    assert b["roc_auc"] is not None


# ──────────────────────────────────────────────
# compute_metric_bundle: multiclass_classification
# ──────────────────────────────────────────────


def test_bundle_multiclass_without_prob():
    y_true = np.array(["a", "b", "c", "a", "b", "c"])
    y_pred = np.array(["a", "b", "a", "a", "b", "b"])
    b = compute_metric_bundle(
        y_true, y_pred, task_family="multiclass_classification"
    )
    assert set(b) == {
        "accuracy", "f1_macro", "f1_weighted", "balanced_accuracy",
        "log_loss", "roc_auc_ovr_macro",
    }
    assert b["accuracy"] is not None
    assert b["f1_macro"] is not None
    assert b["log_loss"] is None
    assert b["roc_auc_ovr_macro"] is None


def test_bundle_multiclass_with_prob():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.2, 0.5, 0.3],
        [0.2, 0.2, 0.6],
    ])
    y_pred = np.argmax(y_prob, axis=1)
    b = compute_metric_bundle(
        y_true, y_pred, y_prob, task_family="multiclass_classification"
    )
    assert b["log_loss"] is not None
    assert b["log_loss"] >= 0
    assert b["roc_auc_ovr_macro"] is not None
    assert 0 <= b["roc_auc_ovr_macro"] <= 1


# ──────────────────────────────────────────────
# compute_metric_bundle: regression
# ──────────────────────────────────────────────


def test_bundle_regression():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 100, 50)
    y_pred = y_true + rng.normal(0, 5, 50)
    b = compute_metric_bundle(y_true, y_pred, task_family="regression")
    assert set(b) == {
        "mae", "mse", "rmse", "r2", "mape", "median_absolute_error",
    }
    assert b["mae"] is not None
    assert b["rmse"] is not None
    assert b["rmse"] >= 0
    assert b["mae"] <= b["rmse"] + 1e-9
    assert b["r2"] is not None
    assert b["r2"] <= 1.0


def test_bundle_regression_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = compute_metric_bundle(y, y, task_family="regression")
    assert b["mae"] == pytest.approx(0.0)
    assert b["mse"] == pytest.approx(0.0)
    assert b["rmse"] == pytest.approx(0.0)
    assert b["r2"] == pytest.approx(1.0)


# ──────────────────────────────────────────────
# compute_metric_bundle: multilabel_classification
# ──────────────────────────────────────────────


def test_bundle_multilabel_without_prob():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(40, 4))
    y_pred = rng.integers(0, 2, size=(40, 4))
    b = compute_metric_bundle(
        y_true, y_pred,
        task_family="multilabel_classification",
        target_mode="multi_label",
    )
    assert set(b) >= {
        "hamming_loss", "f1_micro", "f1_macro", "f1_weighted",
        "subset_accuracy", "roc_auc_macro_ovr",
    }
    assert b["hamming_loss"] is not None
    assert 0 <= b["hamming_loss"] <= 1
    assert b["roc_auc_macro_ovr"] is None


def test_bundle_multilabel_with_prob_matrix():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=(40, 4))
    y_prob = rng.uniform(0, 1, size=(40, 4))
    y_pred = (y_prob > 0.5).astype(int)
    b = compute_metric_bundle(
        y_true, y_pred, y_prob,
        task_family="multilabel_classification",
        target_mode="multi_label",
    )
    assert b["roc_auc_macro_ovr"] is not None
    assert 0 <= b["roc_auc_macro_ovr"] <= 1


def test_bundle_multilabel_with_prob_list_from_multioutput_classifier():
    rng = np.random.default_rng(2)
    n_samples, n_labels = 30, 3
    y_true = rng.integers(0, 2, size=(n_samples, n_labels))
    # Simulate sklearn MultiOutputClassifier.predict_proba format: list of 2D arrays
    y_prob_list = [
        np.column_stack([
            1 - rng.uniform(0, 1, n_samples),
            rng.uniform(0, 1, n_samples),
        ])
        for _ in range(n_labels)
    ]
    y_pred = np.column_stack([np.argmax(m, axis=1) for m in y_prob_list])
    b = compute_metric_bundle(
        y_true, y_pred, y_prob_list,
        task_family="multilabel_classification",
        target_mode="multi_label",
    )
    assert b["roc_auc_macro_ovr"] is not None


# ──────────────────────────────────────────────
# Dispatch resolution
# ──────────────────────────────────────────────


def test_bundle_target_mode_multi_label_overrides_family():
    y_true = np.array([[0, 1], [1, 0], [1, 1]])
    y_pred = np.array([[0, 1], [1, 0], [0, 1]])
    # family says binary but target_mode overrides
    b = compute_metric_bundle(
        y_true, y_pred,
        task_family="binary_classification",
        target_mode="multi_label",
    )
    assert "hamming_loss" in b  # multilabel schema
    assert "roc_auc" not in b   # not binary schema


def test_bundle_coarse_classification_falls_back_to_binary():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    b = compute_metric_bundle(y_true, y_pred, task_family="classification")
    # Coarse "classification" → binary schema (contains accuracy / f1 / roc_auc)
    assert "accuracy" in b
    assert "f1" in b
