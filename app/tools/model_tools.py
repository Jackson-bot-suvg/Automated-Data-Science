"""Model building tools — task-family aware, target_spec aware.

纯函数工具集，供 model_building agent 节点调用。所有函数不依赖 GraphState，
仅基于 DataFrame / sklearn 对象 / target_spec 摘要工作。

支持的任务族与 target_mode 组合：

    task_family                      × target_mode
    ─────────────────────────────────────────────────────────
    binary_classification            × single_target
    multiclass_classification        × single_target
    regression                       × single_target
    multilabel_classification        × multi_label
    regression                       × multi_output  (MultiOutputRegressor)
    clustering                       × single_target (X-only, unsupervised)

对应 ``NODE_CONTRACTS["model_building"]`` 中由 agent 写入的字段与保留工件路径。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from app.utils.task_contract import canonical_candidate_metrics

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Reserved artifact paths (engineering-specification §9)
# ──────────────────────────────────────────────
MODEL_RESULTS_PATH = "artifacts/models/model_results.json"
BEST_MODEL_PATH = "artifacts/models/best_model.pkl"
PREDICTION_PATH = "artifacts/predictions/submission.csv"


# ──────────────────────────────────────────────
# Task family normalization & inference
# ──────────────────────────────────────────────

_TASK_FAMILY_CANONICAL = {
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "clustering",
}


def normalize_task_family(raw: str) -> str:
    """Map coarse/legacy task type strings to the canonical 4-family vocabulary.

    Unknown inputs return "unknown" and trigger downstream inference.
    """
    key = (raw or "").strip().lower()
    if key in _TASK_FAMILY_CANONICAL:
        return key
    if "multilabel" in key or "multi_label" in key or "multi-label" in key:
        return "multilabel_classification"
    if "multiclass" in key or "multi_class" in key or "multi-class" in key:
        return "multiclass_classification"
    if "binary" in key:
        return "binary_classification"
    if "regress" in key:
        return "regression"
    if "classif" in key:
        # Coarse "classification" — caller should refine via infer_task_family(y).
        return "classification_unspecified"
    if "cluster" in key:
        return "clustering"
    return "unknown"


def infer_task_family(
    y: pd.Series | pd.DataFrame,
    target_mode: str = "single_target",
) -> str:
    """Infer canonical task_family from y when upstream target_spec is incomplete.

    Priority: target_mode == "multi_label" → multilabel_classification.
    Otherwise: continuous numeric with high cardinality → regression;
    binary unique values → binary_classification; else multiclass_classification.
    """
    if target_mode == "multi_label":
        return "multilabel_classification"

    # Multi-column DataFrame but mode not explicitly multi_label — treat first column.
    if isinstance(y, pd.DataFrame):
        if y.shape[1] > 1:
            # Fallback: each column binary => multilabel; else single_target on col 0.
            if all(y[c].dropna().nunique() <= 2 for c in y.columns):
                return "multilabel_classification"
            y = y.iloc[:, 0]
        else:
            y = y.iloc[:, 0]

    series = pd.Series(y).dropna()
    if series.empty:
        return "unknown"

    unique_count = series.nunique()
    is_float = pd.api.types.is_float_dtype(series)
    is_int = pd.api.types.is_integer_dtype(series)
    is_numeric = is_float or is_int

    if unique_count == 2:
        return "binary_classification"
    if is_numeric and (is_float or unique_count > 20):
        return "regression"
    return "multiclass_classification"


# ──────────────────────────────────────────────
# Default primary metric & scoring resolution
# ──────────────────────────────────────────────

_DEFAULT_PRIMARY_METRIC = {
    "binary_classification": "roc_auc",
    "multiclass_classification": "f1_macro",
    "regression": "neg_root_mean_squared_error",
    "multilabel_classification": "roc_auc_macro_ovr",
    "clustering": "silhouette_score",
}

# sklearn scorer name mapping per family (primary_metric → sklearn scoring string)
_SKLEARN_SCORER = {
    # binary
    ("binary_classification", "roc_auc"): "roc_auc",
    ("binary_classification", "accuracy"): "accuracy",
    ("binary_classification", "f1"): "f1",
    ("binary_classification", "precision"): "precision",
    ("binary_classification", "recall"): "recall",
    ("binary_classification", "average_precision"): "average_precision",
    # multiclass
    ("multiclass_classification", "f1_macro"): "f1_macro",
    ("multiclass_classification", "f1_weighted"): "f1_weighted",
    ("multiclass_classification", "accuracy"): "accuracy",
    ("multiclass_classification", "balanced_accuracy"): "balanced_accuracy",
    ("multiclass_classification", "neg_log_loss"): "neg_log_loss",
    ("multiclass_classification", "roc_auc_ovr_macro"): "roc_auc_ovr",
    # regression
    ("regression", "neg_root_mean_squared_error"): "neg_root_mean_squared_error",
    ("regression", "neg_mean_squared_error"): "neg_mean_squared_error",
    ("regression", "neg_mean_absolute_error"): "neg_mean_absolute_error",
    ("regression", "r2"): "r2",
    # multilabel (sklearn has limited native support; we compute these manually)
    ("multilabel_classification", "roc_auc_macro_ovr"): None,
    ("multilabel_classification", "f1_macro"): "f1_macro",
    ("multilabel_classification", "f1_micro"): "f1_micro",
    ("multilabel_classification", "hamming_loss"): None,
    # clustering (always computed manually via silhouette/calinski_harabasz)
    ("clustering", "silhouette_score"): None,
    ("clustering", "calinski_harabasz"): None,
    ("clustering", "davies_bouldin"): None,
}


def default_primary_metric(task_family: str) -> str:
    return _DEFAULT_PRIMARY_METRIC.get(task_family, "accuracy")


def metric_direction(primary_metric: str) -> Literal["maximize", "minimize"]:
    """Return 'maximize' for scorers where higher is better, else 'minimize'.

    sklearn already uses neg_* prefix so higher is better for those too.
    """
    if primary_metric.startswith("neg_"):
        return "maximize"
    if primary_metric in {"hamming_loss", "log_loss", "mean_squared_error",
                          "mean_absolute_error", "mean_absolute_percentage_error",
                          "rmse", "mse", "mae", "mape"}:
        return "minimize"
    return "maximize"


def resolve_sklearn_scoring(task_family: str, primary_metric: str) -> str | None:
    """Return the sklearn scoring string, or None if we must compute manually."""
    return _SKLEARN_SCORER.get((task_family, primary_metric))


def canonicalize_metric_name(task_family: str, metric_name: str) -> str:
    """Map metric aliases to task-family-safe canonical names."""

    normalized = canonical_candidate_metrics(task_family, [metric_name])
    if normalized:
        return normalized[0]
    return default_primary_metric(task_family)


def sanitize_candidate_metrics(
    task_family: str,
    metrics: list[str] | None,
) -> list[str]:
    """Return a de-duplicated, task-family-safe metric list."""

    return canonical_candidate_metrics(task_family, metrics or [])


# ──────────────────────────────────────────────
# Feature type detection (unchanged semantics, extra 'exclude')
# ──────────────────────────────────────────────

def detect_feature_types(
    X: pd.DataFrame,
    selected_features: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split columns by dtype; bool counts as numeric.

    Args:
        X: Feature matrix.
        selected_features: Explicit subset preference; None/empty falls back to all.
        exclude: Columns to never include (e.g. target or id columns).

    Returns:
        (num_features, cat_features).
    """
    exclude_set = set(exclude or [])
    base = selected_features or X.columns.tolist()
    cols = [c for c in base if c in X.columns and c not in exclude_set]
    if not cols:
        cols = [c for c in X.columns if c not in exclude_set]

    num_features = [
        c for c in cols
        if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
    ]
    cat_features = [c for c in cols if c not in num_features]
    return num_features, cat_features


# ──────────────────────────────────────────────
# Preprocessor builder (parameterized)
# ──────────────────────────────────────────────

_NUMERIC_IMPUTE_STRATEGIES = {"median", "mean", "most_frequent", "constant"}
_CAT_IMPUTE_STRATEGIES = {"most_frequent", "constant"}
_CAT_ENCODERS = {"onehot", "ordinal"}
_SCALERS = {"standard", "minmax", "robust", "none"}

# LLM planners commonly emit non-standard impute synonyms ("mode" for
# most_frequent, "zero" for constant, etc.). Map them to sklearn-supported
# strategies so a noisy plan does not crash the modeling phase.
_IMPUTE_STRATEGY_ALIASES = {
    "mode": "most_frequent",
    "modal": "most_frequent",
    "most_common": "most_frequent",
    "zero": "constant",
    "fill_zero": "constant",
    "average": "mean",
    "avg": "mean",
}


def build_preprocessor(
    num_features: list[str],
    cat_features: list[str],
    *,
    numeric_impute: str = "median",
    scaler: str = "standard",
    cat_encode: str = "onehot",
    cat_impute: str = "most_frequent",
    onehot_max_cats: int | None = 25,
):
    """Build a ColumnTransformer with configurable steps.

    All non-standard parameters (e.g. scaler='none') are sanitized to safe defaults.

    ``onehot_max_cats`` defaults to 25 (aligned with model_building defaults) as a
    safety cap against cardinality explosions. Set to None to disable the cap.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        RobustScaler,
        StandardScaler,
    )

    # Defensive: upstream (LLM planner) occasionally supplies dict / non-str
    # values such as {"strategy": "median"}. Coerce to expected primitive types
    # before set-membership checks to avoid TypeError: unhashable type.
    if isinstance(numeric_impute, str):
        numeric_impute = _IMPUTE_STRATEGY_ALIASES.get(numeric_impute, numeric_impute)
    if not isinstance(numeric_impute, str) or numeric_impute not in _NUMERIC_IMPUTE_STRATEGIES:
        numeric_impute = "median"
    if not isinstance(scaler, str) or scaler not in _SCALERS:
        scaler = "standard"
    if not isinstance(cat_encode, str) or cat_encode not in _CAT_ENCODERS:
        cat_encode = "onehot"
    if isinstance(cat_impute, str):
        cat_impute = _IMPUTE_STRATEGY_ALIASES.get(cat_impute, cat_impute)
    if not isinstance(cat_impute, str) or cat_impute not in _CAT_IMPUTE_STRATEGIES:
        cat_impute = "most_frequent"
    if onehot_max_cats is not None and not isinstance(onehot_max_cats, int):
        try:
            onehot_max_cats = int(onehot_max_cats)
        except (TypeError, ValueError):
            onehot_max_cats = 25

    transformers = []
    if num_features:
        steps = [("imputer", SimpleImputer(strategy=numeric_impute))]
        if scaler == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            steps.append(("scaler", RobustScaler()))
        # "none" → skip scaler
        transformers.append(("num", Pipeline(steps), num_features))

    if cat_features:
        if cat_encode == "onehot":
            kwargs = {"handle_unknown": "ignore", "sparse_output": False}
            if onehot_max_cats is not None:
                kwargs["max_categories"] = onehot_max_cats
            encoder = OneHotEncoder(**kwargs)
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=cat_impute)),
            ("encoder", encoder),
        ])
        transformers.append(("cat", cat_pipe, cat_features))

    return ColumnTransformer(transformers=transformers)


# ──────────────────────────────────────────────
# XGBoost soft dependency
# ──────────────────────────────────────────────

def _try_import_xgboost():
    """Return (XGBClassifier, XGBRegressor) or (None, None) when missing."""
    try:
        from xgboost import XGBClassifier, XGBRegressor  # type: ignore

        return XGBClassifier, XGBRegressor
    except Exception as exc:  # pragma: no cover
        logger.debug("xgboost not available: %s", exc)
        return None, None


def _try_import_lightgbm():
    """Return (LGBMClassifier, LGBMRegressor) or (None, None) when missing."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

        return LGBMClassifier, LGBMRegressor
    except Exception as exc:  # pragma: no cover
        logger.debug("lightgbm not available: %s", exc)
        return None, None


# ──────────────────────────────────────────────
# Candidate models per task family
# ──────────────────────────────────────────────

def _base_classifier_candidates(
    *,
    prediction_kind: str = "label",
) -> dict[str, dict]:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # class_weight: re-weights per-class loss by inverse frequency. On balanced
    # data the weights collapse to ~1.0 (no-op); on imbalanced data it lifts
    # minority-class signal. Universally safe for AUC (rank-based) — applied
    # by default to LR / RF / SVC. HistGradientBoostingClassifier does not expose
    # class_weight (sklearn API limitation); XGBoost has its own scale_pos_weight
    # path that is awkward to set under MultiOutput wrapping, left as-is.
    cands: dict[str, dict] = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced",
            ),
            "param_grid": {
                "classifier__C": [0.1, 0.3, 1.0, 3.0],
                "classifier__penalty": ["l2"],
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                random_state=42, n_jobs=1, class_weight="balanced_subsample",
            ),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [8, None],
                "classifier__min_samples_leaf": [1],
            },
        },
        "ExtraTrees": {
            "estimator": ExtraTreesClassifier(
                random_state=42, n_jobs=1, class_weight="balanced",
            ),
            "param_grid": {
                "classifier__n_estimators": [300, 600],
                "classifier__max_features": ["sqrt"],
                "classifier__min_samples_leaf": [1, 2],
            },
        },
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingClassifier(
                random_state=42,
                class_weight="balanced",
            ),
            "param_grid": {
                "classifier__max_iter": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [6],
                "classifier__l2_regularization": [0.0, 0.1],
            },
        },
        "SVC": {
            # Keep SVC on the label-only path. Probability calibration is too
            # expensive for our current task suite and is unnecessary when the
            # submission contract expects hard labels.
            "estimator": SVC(
                probability=False, random_state=42, class_weight="balanced",
            ),
            "param_grid": {
                "classifier__C": [0.1, 1.0],
                "classifier__kernel": ["rbf"],
                "classifier__gamma": ["scale"],
            },
        },
    }
    LGBMClassifier, _ = _try_import_lightgbm()
    if LGBMClassifier is not None:
        cands["LightGBM"] = {
            "estimator": LGBMClassifier(
                random_state=42,
                n_jobs=1,
                class_weight="balanced",
                objective=None,
                verbosity=-1,
                force_col_wise=True,
            ),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__learning_rate": [0.03, 0.07],
                "classifier__num_leaves": [31, 63],
                "classifier__min_child_samples": [20],
            },
        }
    XGBClassifier, _ = _try_import_xgboost()
    if XGBClassifier is not None:
        cands["XGBoost"] = {
            "estimator": XGBClassifier(
                random_state=42, n_jobs=1, verbosity=0,
                tree_method="hist", device="cpu",
            ),
            "param_grid": {
                "classifier__n_estimators": [150],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.03, 0.07],
            },
        }
    if str(prediction_kind).lower() == "probability":
        cands.pop("SVC", None)
    return cands


def _base_regressor_candidates() -> dict[str, dict]:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Lasso, Ridge

    cands: dict[str, dict] = {
        "Ridge": {
            "estimator": Ridge(random_state=42),
            "param_grid": {"classifier__alpha": [0.1, 1.0, 3.0, 10.0]},
        },
        "Lasso": {
            "estimator": Lasso(random_state=42, max_iter=10000),
            "param_grid": {"classifier__alpha": [0.03, 0.1, 1.0]},
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=42, n_jobs=1),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [8, None],
                "classifier__min_samples_leaf": [1],
            },
        },
        "ExtraTrees": {
            "estimator": ExtraTreesRegressor(random_state=42, n_jobs=1),
            "param_grid": {
                "classifier__n_estimators": [300, 600],
                "classifier__max_features": ["sqrt"],
                "classifier__min_samples_leaf": [1, 2],
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__learning_rate": [0.03, 0.07],
                "classifier__max_depth": [2, 3],
            },
        },
    }
    _, LGBMRegressor = _try_import_lightgbm()
    if LGBMRegressor is not None:
        cands["LightGBM"] = {
            "estimator": LGBMRegressor(
                random_state=42,
                n_jobs=1,
                objective="regression",
                verbosity=-1,
                force_col_wise=True,
            ),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__learning_rate": [0.03, 0.07],
                "classifier__num_leaves": [31, 63],
                "classifier__min_child_samples": [20],
            },
        }
    _, XGBRegressor = _try_import_xgboost()
    if XGBRegressor is not None:
        cands["XGBoost"] = {
            "estimator": XGBRegressor(
                random_state=42, n_jobs=1, verbosity=0,
                tree_method="hist", device="cpu",
            ),
            "param_grid": {
                "classifier__n_estimators": [150],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.03, 0.07],
            },
        }
    return cands


def _wrap_candidates_for_multi_label(base: dict[str, dict]) -> dict[str, dict]:
    """Wrap single-output classifiers in MultiOutputClassifier for multi-label tasks.

    Rewrites param_grid keys: classifier__C → classifier__estimator__C.
    """
    from sklearn.multioutput import MultiOutputClassifier

    wrapped: dict[str, dict] = {}
    for name, cfg in base.items():
        inner = cfg["estimator"]
        new_grid = {
            key.replace("classifier__", "classifier__estimator__", 1): value
            for key, value in cfg["param_grid"].items()
        }
        wrapped[f"MultiOutput_{name}"] = {
            "estimator": MultiOutputClassifier(inner, n_jobs=1),
            "param_grid": new_grid,
        }
    return wrapped


def _wrap_candidates_for_multi_output_regression(base: dict[str, dict]) -> dict[str, dict]:
    """Wrap single-output regressors in MultiOutputRegressor for multi-output tasks.

    Rewrites param_grid keys: classifier__alpha → classifier__estimator__alpha.
    """
    from sklearn.multioutput import MultiOutputRegressor

    wrapped: dict[str, dict] = {}
    for name, cfg in base.items():
        inner = cfg["estimator"]
        new_grid = {
            key.replace("classifier__", "classifier__estimator__", 1): value
            for key, value in cfg["param_grid"].items()
        }
        wrapped[f"MultiOutput_{name}"] = {
            "estimator": MultiOutputRegressor(inner, n_jobs=1),
            "param_grid": new_grid,
        }
    return wrapped


def _cluster_candidates() -> dict[str, dict]:
    """Return unsupervised clustering candidate configs.

    param_grid keys are direct estimator params (no Pipeline prefix) because
    ``fit_and_select_clustering_model`` handles the grid manually.
    """
    from sklearn.cluster import KMeans

    cands: dict[str, dict] = {
        "KMeans": {
            "estimator": KMeans(random_state=42, n_init="auto"),
            "param_grid": {"n_clusters": [3, 5, 8]},
        },
    }
    try:
        from sklearn.cluster import DBSCAN  # type: ignore

        cands["DBSCAN"] = {
            "estimator": DBSCAN(),
            "param_grid": {"eps": [0.3, 0.5, 1.0], "min_samples": [3, 5]},
        }
    except Exception:  # pragma: no cover
        pass
    return cands


def _subset_candidates(base: dict[str, dict], allowed_names: tuple[str, ...]) -> dict[str, dict]:
    """Keep a stable subset of candidate estimators by name."""

    return {
        name: cfg
        for name, cfg in base.items()
        if name in allowed_names
    }


def _param_grid_combo_count(param_grid: dict[str, Any]) -> int:
    """Return the number of grid combinations without materializing them."""

    total = 1
    for value in (param_grid or {}).values():
        if isinstance(value, list):
            total *= max(1, len(value))
        else:
            total *= 1
    return total


def _copy_candidate_with_grid(cfg: dict[str, Any], grid: dict[str, list[Any]]) -> dict[str, Any]:
    return {"estimator": cfg["estimator"], "param_grid": grid}


def optimize_candidate_search_space(
    candidates: dict[str, dict],
    *,
    n_samples: int,
    n_features: int,
    task_family: str,
) -> tuple[dict[str, dict], list[str]]:
    """Apply generic compute-budget rules to candidate grids.

    The rules are intentionally data-size based, not dataset-name based:
    large tabular datasets keep the strongest fast boosters and use narrow grids
    that include the empirically good defaults. Smaller datasets retain the
    broader candidate family where RF/ExtraTrees can still be useful.
    """

    family = normalize_task_family(task_family)
    notes: list[str] = []
    if not candidates:
        return candidates, notes

    is_large = n_samples >= 50_000 or (n_samples * max(1, n_features)) >= 1_000_000
    is_medium = not is_large and n_samples >= 10_000
    optimized = dict(candidates)

    if is_large and family in {"binary_classification", "multiclass_classification"}:
        preferred_names = ("LogisticRegression", "HistGradientBoosting", "LightGBM", "XGBoost")
        optimized = _subset_candidates(optimized, preferred_names)
        notes.append(
            "model_search_budget: large tabular dataset -> skip high-cost RF/ExtraTrees grid search"
        )
        if "LogisticRegression" in optimized:
            optimized["LogisticRegression"] = _copy_candidate_with_grid(
                optimized["LogisticRegression"],
                {
                    "classifier__C": [3.0],
                    "classifier__penalty": ["l2"],
                },
            )
        if "HistGradientBoosting" in optimized:
            optimized["HistGradientBoosting"] = _copy_candidate_with_grid(
                optimized["HistGradientBoosting"],
                {
                    "classifier__max_iter": [200],
                    "classifier__learning_rate": [0.05],
                    "classifier__max_depth": [6],
                    "classifier__l2_regularization": [0.0],
                },
            )
        if "LightGBM" in optimized:
            optimized["LightGBM"] = _copy_candidate_with_grid(
                optimized["LightGBM"],
                {
                    "classifier__n_estimators": [400],
                    "classifier__learning_rate": [0.03, 0.07],
                    "classifier__num_leaves": [31],
                    "classifier__min_child_samples": [20],
                },
            )
        if "XGBoost" in optimized:
            optimized["XGBoost"] = _copy_candidate_with_grid(
                optimized["XGBoost"],
                {
                    "classifier__n_estimators": [150],
                    "classifier__max_depth": [3],
                    "classifier__learning_rate": [0.05],
                },
            )
        notes.append(
            "model_search_budget: large tabular dataset -> narrow grids around fast boosting defaults"
        )
    elif is_large and family == "regression":
        preferred_names = ("Ridge", "GradientBoosting", "LightGBM", "XGBoost")
        optimized = _subset_candidates(optimized, preferred_names)
        notes.append(
            "model_search_budget: large regression dataset -> skip high-cost RF/ExtraTrees grid search"
        )
        if "Ridge" in optimized:
            optimized["Ridge"] = _copy_candidate_with_grid(
                optimized["Ridge"],
                {"classifier__alpha": [1.0, 3.0, 10.0]},
            )
        if "GradientBoosting" in optimized:
            optimized["GradientBoosting"] = _copy_candidate_with_grid(
                optimized["GradientBoosting"],
                {
                    "classifier__n_estimators": [200],
                    "classifier__learning_rate": [0.05],
                    "classifier__max_depth": [3],
                },
            )
        if "LightGBM" in optimized:
            optimized["LightGBM"] = _copy_candidate_with_grid(
                optimized["LightGBM"],
                {
                    "classifier__n_estimators": [400],
                    "classifier__learning_rate": [0.03, 0.07],
                    "classifier__num_leaves": [31],
                    "classifier__min_child_samples": [20],
                },
            )
    elif is_medium:
        for name in ("RandomForest", "ExtraTrees"):
            if name in optimized:
                grid = optimized[name].get("param_grid") or {}
                trimmed = {
                    key: (value[:1] if isinstance(value, list) else [value])
                    for key, value in grid.items()
                }
                optimized[name] = _copy_candidate_with_grid(optimized[name], trimmed)
        notes.append(
            "model_search_budget: medium dataset -> single-point RF/ExtraTrees screening"
        )

    before_combos = sum(_param_grid_combo_count(cfg.get("param_grid", {})) for cfg in candidates.values())
    after_combos = sum(_param_grid_combo_count(cfg.get("param_grid", {})) for cfg in optimized.values())
    if after_combos < before_combos:
        notes.append(
            f"model_search_budget: grid combinations reduced from {before_combos} to {after_combos}"
        )
    return optimized, notes


def get_candidate_models(
    task_family: str = "binary_classification",
    target_mode: str = "single_target",
    prediction_kind: str = "label",
) -> dict[str, dict]:
    """Return task-family and target-mode aware candidate estimator dict.

    Returned dict has shape:
        {name: {"estimator": sklearn estimator, "param_grid": dict}}

    For clustering, returns candidates for ``fit_and_select_clustering_model``
    (param_grid keys are direct estimator params, not pipeline-prefixed).
    ``prediction_kind`` is used to suppress candidates such as SVC when the
    submission contract requires probabilities.
    """
    family = normalize_task_family(task_family)

    if family == "clustering":
        return _cluster_candidates()

    if family == "regression":
        if target_mode == "multi_output":
            # Wrap regressors in MultiOutputRegressor; exclude Lasso (rarely best for multi-output).
            base = _subset_candidates(
                _base_regressor_candidates(),
                ("Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "LightGBM", "XGBoost"),
            )
            return _wrap_candidates_for_multi_output_regression(base)
        return _base_regressor_candidates()

    if family == "multilabel_classification" or target_mode == "multi_label":
        # SVC excluded: probability=True + MultiOutputClassifier multiplies Platt-scaling cost by n_labels.
        base = _subset_candidates(
            _base_classifier_candidates(prediction_kind=prediction_kind),
            ("LogisticRegression", "RandomForest", "ExtraTrees", "LightGBM", "XGBoost"),
        )
        return _wrap_candidates_for_multi_label(base)

    if family in {"binary_classification", "multiclass_classification",
                  "classification_unspecified", "unknown"}:
        return _base_classifier_candidates(prediction_kind=prediction_kind)

    # Fallback
    return _base_classifier_candidates()


# ──────────────────────────────────────────────
# Cross-validation strategy
# ──────────────────────────────────────────────

_STRATIFIED_COMPATIBLE_FAMILIES = {"binary_classification", "multiclass_classification"}


def build_cv(
    task_family: str,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    strategy: str | None = None,
):
    """Build a sklearn CV splitter appropriate for the task family.

    Behavior:
        - classification (binary/multiclass): StratifiedKFold by default.
        - regression / multilabel_classification / multi-output regression /
          clustering / unknown: plain KFold by default.
        - When ``strategy`` is supplied, honor it ONLY if compatible with the
          task family. ``stratified_kfold`` is silently downgraded to KFold
          for any family outside ``_STRATIFIED_COMPATIBLE_FAMILIES`` because
          sklearn's StratifiedKFold rejects multilabel-indicator targets.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    family = normalize_task_family(task_family)
    requested = (strategy or "").lower()

    if requested == "stratified_kfold" and family not in _STRATIFIED_COMPATIBLE_FAMILIES:
        logger.info(
            "build_cv: downgrading stratified_kfold→kfold for task_family=%s "
            "(StratifiedKFold rejects multilabel-indicator / continuous targets)",
            family,
        )
        requested = "kfold"

    if requested == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if requested == "stratified_kfold":
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    if family in _STRATIFIED_COMPATIBLE_FAMILIES:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    # regression / multilabel_classification / multi_output / clustering / unknown → KFold
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ──────────────────────────────────────────────
# Fit + select (task-family aware scoring & refit)
# ──────────────────────────────────────────────

def _score_family_secondary(task_family: str) -> list[str]:
    """Default secondary sklearn scorers per family (for cross_val_score reporting)."""
    family = normalize_task_family(task_family)
    if family == "binary_classification":
        return ["accuracy", "f1", "roc_auc"]
    if family == "multiclass_classification":
        return ["accuracy", "f1_macro", "balanced_accuracy"]
    if family == "regression":
        return ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    if family == "multilabel_classification":
        return ["f1_micro", "f1_macro"]
    return ["accuracy"]


def _resolve_candidate_parallelism(
    candidate_count: int,
    *,
    n_samples: int | None = None,
    total_param_combos: int | None = None,
) -> int:
    """Return a safe candidate-level parallelism for model selection.

    We keep GridSearchCV itself on ``n_jobs=1`` and parallelize across
    candidate models instead. This avoids nested joblib oversubscription while
    still using the available cores on the machine.
    """
    if candidate_count <= 1:
        return 1

    requested = os.getenv("MODEL_BUILDING_MAX_PARALLEL_CANDIDATES", "").strip()
    if requested:
        try:
            return max(1, min(candidate_count, int(requested)))
        except ValueError:
            logger.warning(
                "Ignoring invalid MODEL_BUILDING_MAX_PARALLEL_CANDIDATES=%r",
                requested,
            )

    cpu_count = os.cpu_count() or 1
    if (n_samples or 0) >= 50_000:
        return 1
    if (total_param_combos or 0) >= 20:
        return max(1, min(candidate_count, cpu_count, 2))
    return max(1, min(candidate_count, cpu_count, 4))


def _fit_single_candidate(
    name: str,
    cfg: dict[str, Any],
    *,
    X_train: pd.DataFrame,
    y_train,
    preprocessor,
    cv,
    refit_scoring,
    custom_scorers: dict[str, Any],
    sklearn_scoring: str | None,
) -> tuple[str, dict[str, Any], float | None, Any]:
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    cache_dir = tempfile.mkdtemp(prefix="model-preprocess-cache-")
    memory = joblib.Memory(cache_dir, verbose=0)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", cfg["estimator"]),
    ], memory=memory)
    try:
        gs = GridSearchCV(
            pipe,
            param_grid=cfg["param_grid"],
            cv=cv,
            scoring=refit_scoring,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        gs.fit(X_train, y_train)
        best_est = gs.best_estimator_

        if isinstance(refit_scoring, str):
            refit_key = refit_scoring
        else:
            refit_key = next(
                (k for k, v in custom_scorers.items() if v is refit_scoring),
                "custom",
            )

        result: dict[str, Any] = {
            "best_params": _strip_param_prefixes(gs.best_params_),
            f"cv_{refit_key}_mean": round(float(gs.best_score_), 4),
        }
        primary_key = (
            f"cv_{sklearn_scoring}_mean" if sklearn_scoring else f"cv_{refit_key}_mean"
        )
        candidate_score = float(result.get(primary_key, gs.best_score_))
        return name, result, candidate_score, best_est
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def _make_per_target_macro_f1_scorer():
    """Return a sklearn-compatible scorer averaging f1_macro across each target column.

    Used for multiclass-multioutput targets where built-in 'f1_micro' / 'f1_macro'
    rejects the label type. Operates on (n_samples, n_targets) y arrays.
    """
    from sklearn.metrics import f1_score, make_scorer

    def _per_target_macro_f1(y_true, y_pred):
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)
        cols = min(y_true_arr.shape[1], y_pred_arr.shape[1])
        scores = []
        for j in range(cols):
            scores.append(
                f1_score(
                    y_true_arr[:, j],
                    y_pred_arr[:, j],
                    average="macro",
                    zero_division=0,
                )
            )
        return float(np.mean(scores)) if scores else 0.0

    return make_scorer(_per_target_macro_f1)


def _make_subset_accuracy_scorer():
    """Subset-accuracy scorer that accepts both multilabel-indicator and
    multiclass-multioutput y (sklearn's built-in 'accuracy' rejects the latter)."""
    from sklearn.metrics import make_scorer

    def _subset_accuracy(y_true, y_pred):
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)
        return float(np.mean(np.all(y_true_arr == y_pred_arr, axis=1)))

    return make_scorer(_subset_accuracy)


def _make_per_target_roc_auc_scorer():
    """Per-target ROC-AUC averaged across columns (multilabel-indicator y).

    Built-in sklearn string scorer 'roc_auc' / 'roc_auc_ovr' rejects
    multilabel-indicator y. We wrap roc_auc_score with make_scorer and use
    response_method='predict_proba' so MultiOutputClassifier's list-of-arrays
    proba is normalized to an (n_samples, n_targets) positive-class matrix
    before scoring. Columns where y_true is constant in a fold (no positives
    or no negatives) are skipped — sklearn would otherwise raise.
    """
    from sklearn.metrics import make_scorer, roc_auc_score

    def _per_target_auc(y_true, y_prob):
        y_true_arr = np.asarray(y_true)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
        prob_matrix = _multi_label_positive_prob_matrix(y_prob)
        if prob_matrix is None:
            return 0.0
        cols = min(y_true_arr.shape[1], prob_matrix.shape[1])
        scores: list[float] = []
        for j in range(cols):
            yt = y_true_arr[:, j]
            unique = np.unique(yt[~pd.isna(yt)] if hasattr(pd, "isna") else yt)
            if len(unique) < 2:
                # Single-class column in this fold: AUC undefined; skip.
                continue
            try:
                scores.append(float(roc_auc_score(yt, prob_matrix[:, j])))
            except Exception:  # noqa: BLE001
                continue
        return float(np.mean(scores)) if scores else 0.0

    return make_scorer(
        _per_target_auc,
        response_method="predict_proba",
        greater_is_better=True,
    )


def _score_holdout_predictions(
    *,
    y_true,
    y_pred,
    y_prob,
    task_family: str,
    primary_metric: str,
) -> dict[str, float]:
    """Compute holdout metrics for the no-CV multilabel path."""

    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

    family = normalize_task_family(task_family)
    metrics: dict[str, float] = {}

    if family == "multilabel_classification":
        if primary_metric == "roc_auc_macro_ovr":
            prob_matrix = _multi_label_positive_prob_matrix(y_prob)
            y_true_arr = np.asarray(y_true)
            if y_true_arr.ndim == 1:
                y_true_arr = y_true_arr.reshape(-1, 1)
            cols = min(y_true_arr.shape[1], prob_matrix.shape[1])
            scores: list[float] = []
            for index in range(cols):
                yt = y_true_arr[:, index]
                if len(np.unique(yt[~pd.isna(yt)])) < 2:
                    continue
                scores.append(float(roc_auc_score(yt, prob_matrix[:, index])))
            score = float(np.mean(scores)) if scores else 0.0
            metrics["val_roc_auc_macro_ovr"] = round(score, 4)
            metrics["val_per_target_auc_mean"] = round(score, 4)
            return metrics
        if primary_metric in {"f1_macro", "f1_micro"}:
            average = "micro" if primary_metric == "f1_micro" else "macro"
            score = float(f1_score(y_true, y_pred, average=average, zero_division=0))
            metrics[f"val_{primary_metric}"] = round(score, 4)
            return metrics

    if family == "binary_classification" and primary_metric == "roc_auc":
        prob_arr = np.asarray(y_prob)
        if prob_arr.ndim == 2 and prob_arr.shape[1] >= 2:
            score_values = prob_arr[:, 1]
        else:
            score_values = prob_arr.ravel()
        metrics["val_roc_auc"] = round(float(roc_auc_score(y_true, score_values)), 4)
        return metrics
    if family in {"binary_classification", "multiclass_classification"}:
        if primary_metric == "accuracy":
            metrics["val_accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
            return metrics
        if primary_metric == "f1_macro":
            metrics["val_f1_macro"] = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
            return metrics
    if family == "regression" and primary_metric == "neg_root_mean_squared_error":
        mse = mean_squared_error(y_true, y_pred)
        metrics["val_neg_root_mean_squared_error"] = round(-float(np.sqrt(mse)), 4)
        return metrics

    # Last-resort metric for unsupported combinations.
    metrics[f"val_{primary_metric or 'score'}"] = round(float(accuracy_score(y_true, y_pred)), 4)
    return metrics


def _fit_and_select_holdout_model(
    X_train: pd.DataFrame,
    y_train,
    preprocessor,
    candidates: dict[str, dict],
    *,
    task_family: str,
    primary_metric: str,
    direction: Literal["maximize", "minimize"],
    random_state: int,
) -> tuple[Any, str, dict[str, dict[str, Any]]]:
    """Select models on a fixed holdout split, then refit the winner on full data."""

    from sklearn.base import clone
    from sklearn.model_selection import ParameterGrid, train_test_split
    from sklearn.pipeline import Pipeline

    y_for_split = y_train
    stratify = None
    # Multilabel indicators are not accepted by stratify; single-target callers
    # may still benefit from stratification if this helper is reused later.
    if not isinstance(y_train, pd.DataFrame):
        try:
            counts = pd.Series(y_train).value_counts(dropna=False)
            if len(counts) > 1 and int(counts.min()) >= 2:
                stratify = y_train
        except Exception:
            stratify = None

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_for_split,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    all_results: dict[str, dict[str, Any]] = {}
    best_name = ""
    best_score = float("-inf") if direction == "maximize" else float("inf")
    best_params: dict[str, Any] = {}

    def _is_better(candidate_score: float) -> bool:
        if direction == "maximize":
            return candidate_score > best_score
        return candidate_score < best_score

    for name, cfg in candidates.items():
        candidate_best_score = float("-inf") if direction == "maximize" else float("inf")
        candidate_best_params: dict[str, Any] = {}
        candidate_best_metrics: dict[str, float] = {}
        try:
            param_grid = cfg.get("param_grid") or {}
            for params in ParameterGrid(param_grid or {}):
                pipe = Pipeline([
                    ("preprocessor", clone(preprocessor)),
                    ("classifier", clone(cfg["estimator"])),
                ])
                if params:
                    pipe.set_params(**params)
                pipe.fit(X_fit, y_fit)
                y_pred = pipe.predict(X_valid)
                try:
                    y_prob = pipe.predict_proba(X_valid)
                except Exception:  # noqa: BLE001
                    y_prob = None
                metrics = _score_holdout_predictions(
                    y_true=y_valid,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    task_family=task_family,
                    primary_metric=primary_metric,
                )
                score = float(next(iter(metrics.values())))
                if (
                    (direction == "maximize" and score > candidate_best_score)
                    or (direction == "minimize" and score < candidate_best_score)
                ):
                    candidate_best_score = score
                    candidate_best_params = dict(params)
                    candidate_best_metrics = metrics

            if not candidate_best_metrics:
                raise ValueError("holdout validation produced no metric")
            all_results[name] = {
                "best_params": _strip_param_prefixes(candidate_best_params),
                **candidate_best_metrics,
                "_selection": "holdout",
            }
            if _is_better(candidate_best_score):
                best_score = candidate_best_score
                best_name = name
                best_params = candidate_best_params
        except Exception as exc:  # noqa: BLE001
            all_results[name] = {"error": str(exc)}

    if not best_name:
        return None, "", all_results

    best_cfg = candidates[best_name]
    best_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(best_cfg["estimator"])),
    ])
    if best_params:
        best_pipeline.set_params(**best_params)
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, best_name, all_results


def fit_and_select_model(
    X_train: pd.DataFrame,
    y_train,
    preprocessor,
    candidates: dict[str, dict],
    *,
    task_family: str = "binary_classification",
    target_mode: str = "single_target",
    cv=None,
    primary_metric: str | None = None,
    secondary_metrics: list[str] | None = None,
    random_state: int = 42,
    multi_target_subtype: str | None = None,
) -> tuple[Any, str, dict[str, dict[str, Any]]]:
    """Run GridSearchCV per candidate, pick the best by primary_metric direction.

    Args:
        X_train: Feature matrix.
        y_train: Series (single-target) or DataFrame (multi-label).
        preprocessor: ColumnTransformer.
        candidates: {name: {"estimator", "param_grid"}}.
        task_family: canonical family string.
        target_mode: single_target / multi_label / multi_output.
        cv: sklearn CV splitter; if None, built from task_family.
        primary_metric: scorer for GridSearchCV refit; default per family.
        secondary_metrics: accepted for compatibility but not re-scored after
            the primary-metric grid search.

    Returns:
        (best_pipeline, best_name, all_results)
        - best_pipeline: refitted sklearn Pipeline, or None if every candidate failed.
        - best_name: name of the winning candidate, "" on total failure.
        - all_results: {name: {best_params, cv_<metric>_mean/std, ...} | {error}}.
    """
    family = normalize_task_family(task_family)
    if primary_metric is None:
        primary_metric = default_primary_metric(family)
    primary_metric = canonicalize_metric_name(family, primary_metric)
    direction = metric_direction(primary_metric)
    sklearn_scoring = resolve_sklearn_scoring(family, primary_metric)
    if cv is None:
        cv = build_cv(family, random_state=random_state)

    # Fallback scoring when primary isn't a native sklearn scorer (e.g. multilabel AUC).
    refit_scoring = sklearn_scoring or resolve_sklearn_scoring(
        family,
        default_primary_metric(family),
    )
    if refit_scoring is None:
        refit_scoring = "f1_micro" if family == "multilabel_classification" else "accuracy"

    custom_scorers: dict[str, Any] = {}

    # Multilabel + AUC primary: sklearn has no built-in string scorer that
    # accepts multilabel-indicator y, so the fallback above silently degrades
    # to f1_micro. Wrap roc_auc_score with make_scorer so GridSearchCV actually
    # optimizes the metric the upstream agents asked for.
    multi_target_subtype = (multi_target_subtype or "").strip()
    if (
        family == "multilabel_classification"
        and multi_target_subtype != "multiclass_multioutput"
        and primary_metric == "roc_auc_macro_ovr"
    ):
        per_target_auc_scorer = _make_per_target_roc_auc_scorer()
        refit_scoring = per_target_auc_scorer
        custom_scorers["per_target_auc"] = per_target_auc_scorer
        sklearn_scoring = None
        direction = "maximize"

    if family == "multilabel_classification" or target_mode == "multi_label":
        return _fit_and_select_holdout_model(
            X_train,
            y_train,
            preprocessor,
            candidates,
            task_family=family,
            primary_metric=primary_metric,
            direction=direction,
            random_state=random_state,
        )

    # Multiclass-multioutput targets cannot use built-in multilabel scorers
    # (f1_micro / hamming_loss / etc.) — sklearn's check_classification_targets
    # raises before fit completes. Route to subset-accuracy + a per-target
    # macro-F1 mean instead. Binary multilabel preserves the existing path.
    if multi_target_subtype == "multiclass_multioutput":
        # Built-in 'accuracy' / 'f1_*' scorers reject multiclass-multioutput
        # via check_classification_targets. Wrap subset-accuracy + per-target
        # macro-F1 with make_scorer to bypass that check.
        subset_acc_scorer = _make_subset_accuracy_scorer()
        per_target_f1_scorer = _make_per_target_macro_f1_scorer()
        refit_scoring = subset_acc_scorer
        secondary = ["subset_accuracy", "per_target_macro_f1"]
        custom_scorers = {
            "subset_accuracy": subset_acc_scorer,
            "per_target_macro_f1": per_target_f1_scorer,
        }
        sklearn_scoring = None
        direction = "maximize"

    all_results: dict[str, dict[str, Any]] = {}
    best_pipeline = None
    best_name = ""
    best_score = float("-inf") if direction == "maximize" else float("inf")

    def _is_better(candidate_score: float) -> bool:
        if direction == "maximize":
            return candidate_score > best_score
        return candidate_score < best_score

    ordered_names = list(candidates.keys())
    candidate_scores: dict[str, float] = {}
    candidate_pipelines: dict[str, Any] = {}
    total_param_combos = sum(
        _param_grid_combo_count(candidates[name].get("param_grid", {}))
        for name in ordered_names
    )
    max_workers = _resolve_candidate_parallelism(
        len(ordered_names),
        n_samples=len(X_train),
        total_param_combos=total_param_combos,
    )

    if max_workers == 1:
        for name in ordered_names:
            cfg = candidates[name]
            try:
                _, result, candidate_score, best_est = _fit_single_candidate(
                    name,
                    cfg,
                    X_train=X_train,
                    y_train=y_train,
                    preprocessor=preprocessor,
                    cv=cv,
                    refit_scoring=refit_scoring,
                    custom_scorers=custom_scorers,
                    sklearn_scoring=sklearn_scoring,
                )
                all_results[name] = result
                if candidate_score is not None:
                    candidate_scores[name] = candidate_score
                    candidate_pipelines[name] = best_est
            except Exception as exc:  # noqa: BLE001
                all_results[name] = {"error": str(exc)}
    else:
        logger.info(
            "fit_and_select_model: evaluating %d candidates with max_workers=%d",
            len(ordered_names),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _fit_single_candidate,
                    name,
                    candidates[name],
                    X_train=X_train,
                    y_train=y_train,
                    preprocessor=preprocessor,
                    cv=cv,
                    refit_scoring=refit_scoring,
                    custom_scorers=custom_scorers,
                    sklearn_scoring=sklearn_scoring,
                ): name
                for name in ordered_names
            }
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    _, result, candidate_score, best_est = future.result()
                    all_results[name] = result
                    if candidate_score is not None:
                        candidate_scores[name] = candidate_score
                        candidate_pipelines[name] = best_est
                except Exception as exc:  # noqa: BLE001
                    all_results[name] = {"error": str(exc)}

    for name in ordered_names:
        candidate_score = candidate_scores.get(name)
        if candidate_score is None:
            continue
        if _is_better(candidate_score):
            best_score = candidate_score
            best_name = name
            best_pipeline = candidate_pipelines.get(name)

    return best_pipeline, best_name, all_results


def fit_and_select_clustering_model(
    X_train: pd.DataFrame,
    preprocessor,
    candidates: dict[str, dict],
    *,
    random_state: int = 42,
) -> tuple[Any, str, dict[str, dict[str, Any]]]:
    """Fit clustering models without labels; select by silhouette score.

    param_grid keys in ``candidates`` are direct estimator params (no
    ``classifier__`` prefix) — consistent with ``_cluster_candidates()``.

    Returns (best_pipeline, best_name, all_results) using step name
    "clusterer" (not "classifier") inside the pipeline.

    DBSCAN and other algorithms without ``predict()`` can fit but will not
    produce test-set submission predictions.
    """
    from itertools import product as iterproduct

    from sklearn.base import clone
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.pipeline import Pipeline

    all_results: dict[str, dict[str, Any]] = {}
    best_pipeline = None
    best_name = ""
    best_silhouette = float("-inf")

    # Precompute transformed X once (preprocessor is not yet fitted).
    # We will fit preprocessor inside each Pipeline, then use transform for metrics.

    for name, cfg in candidates.items():
        try:
            param_grid = cfg.get("param_grid", {})
            keys = list(param_grid.keys())
            value_lists = [
                v if isinstance(v, list) else [v]
                for v in (param_grid[k] for k in keys)
            ]
            combos = list(iterproduct(*value_lists)) if keys else [()]

            candidate_best_pipe = None
            candidate_best_params: dict[str, Any] = {}
            candidate_best_sil = float("-inf")
            candidate_best_metrics: dict[str, Any] = {}

            for combo in combos:
                params = dict(zip(keys, combo))
                est = clone(cfg["estimator"])
                if params:
                    est.set_params(**params)
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("clusterer", est),
                ])
                pipe.fit(X_train)
                labels = np.asarray(pipe.named_steps["clusterer"].labels_)

                # DBSCAN assigns -1 to noise; exclude from metric computation.
                valid_mask = labels != -1
                n_valid = int(valid_mask.sum())
                n_clusters = int(len(set(labels[valid_mask].tolist())) if n_valid > 0 else 0)

                if n_valid < 10 or n_clusters < 2:
                    continue  # degenerate solution

                X_transformed = pipe.named_steps["preprocessor"].transform(X_train)
                X_valid = X_transformed[valid_mask]
                labels_valid = labels[valid_mask]
                sample_size = min(n_valid, 2000)

                sil = float(
                    silhouette_score(X_valid, labels_valid,
                                     sample_size=sample_size, random_state=random_state)
                )

                if sil > candidate_best_sil:
                    candidate_best_sil = sil
                    candidate_best_pipe = pipe
                    candidate_best_params = params
                    metrics: dict[str, Any] = {
                        "silhouette_score": round(sil, 4),
                        "n_clusters": n_clusters,
                        "n_noise_points": int((~valid_mask).sum()),
                    }
                    try:
                        metrics["calinski_harabasz"] = round(
                            float(calinski_harabasz_score(X_valid, labels_valid)), 4
                        )
                        metrics["davies_bouldin"] = round(
                            float(davies_bouldin_score(X_valid, labels_valid)), 4
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    candidate_best_metrics = metrics

            if candidate_best_pipe is not None:
                all_results[name] = {
                    "best_params": candidate_best_params,
                    **candidate_best_metrics,
                }
                if candidate_best_sil > best_silhouette:
                    best_silhouette = candidate_best_sil
                    best_name = name
                    best_pipeline = candidate_best_pipe
            else:
                all_results[name] = {"error": "all parameter combinations yielded degenerate clusters"}

        except Exception as exc:  # noqa: BLE001
            all_results[name] = {"error": str(exc)}

    return best_pipeline, best_name, all_results


def _strip_param_prefixes(params: dict[str, Any]) -> dict[str, Any]:
    """Strip sklearn pipeline prefix(es) so persisted params are human-readable."""
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        # classifier__estimator__C → estimator__C → C
        stripped = key
        for prefix in ("classifier__", "estimator__"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        cleaned[stripped] = value
    return cleaned


# ──────────────────────────────────────────────
# Prediction & submission (target_spec aware)
# ──────────────────────────────────────────────

def predict_for_submission(
    pipeline,
    X_test: pd.DataFrame,
    *,
    task_family: str,
    target_mode: str = "single_target",
    prediction_kind: str = "label",
    output_columns: list[str] | None = None,
    multi_target_subtype: str = "",
) -> pd.DataFrame:
    """Run pipeline predictions and return a DataFrame with columns=output_columns.

    Dispatch logic:
        - single_target + label: predict
        - single_target + probability (binary): predict_proba[:, 1]
        - single_target + probability (multiclass): predict_proba (matrix) →
              one column per class; callers requesting single column get argmax.
        - multi_label + probability: per-label positive probability (matrix)
        - multi_label + label: predict (matrix of 0/1)
        - regression: predict (continuous)
    """
    family = normalize_task_family(task_family)
    output_columns = list(output_columns or [])

    if family == "clustering":
        # Clustering: assign cluster labels if the estimator supports predict().
        # DBSCAN and similar density-based estimators lack predict(); fall back to
        # returning labels from the training fit (stored on the fitted estimator).
        clusterer = (
            pipeline.named_steps.get("clusterer")
            if hasattr(pipeline, "named_steps") else None
        )
        if clusterer is not None and hasattr(clusterer, "predict"):
            labels = pipeline.predict(X_test)
        elif clusterer is not None and hasattr(clusterer, "labels_"):
            labels = np.asarray(clusterer.labels_)
        else:
            labels = pipeline.predict(X_test)
        col = output_columns[0] if output_columns else "cluster"
        return pd.DataFrame({col: labels})

    if family == "regression":
        preds = pipeline.predict(X_test)
        if preds.ndim == 1:
            col = output_columns[0] if output_columns else "prediction"
            return pd.DataFrame({col: preds})
        # multi_output regression (predicted but not fully supported this version)
        cols = output_columns or [f"y_{i}" for i in range(preds.shape[1])]
        return pd.DataFrame(preds, columns=cols)

    if family == "multilabel_classification" or target_mode == "multi_label":
        # For multiclass-multioutput, "positive class probability per label" is
        # ill-defined; emit hard labels instead so the submission is interpretable.
        wants_probability = (
            prediction_kind == "probability"
            and multi_target_subtype != "multiclass_multioutput"
        )
        if wants_probability and hasattr(pipeline, "predict_proba"):
            raw = pipeline.predict_proba(X_test)
            proba = _multi_label_positive_prob_matrix(raw)
            cols = output_columns or [f"label_{i}" for i in range(proba.shape[1])]
            # Guard against width mismatch
            if proba.shape[1] != len(cols):
                cols = cols[: proba.shape[1]] + [
                    f"label_{i}" for i in range(len(cols), proba.shape[1])
                ]
            return pd.DataFrame(proba, columns=cols)
        preds = pipeline.predict(X_test)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        cols = output_columns or [f"label_{i}" for i in range(preds.shape[1])]
        return pd.DataFrame(preds, columns=cols)

    # Binary / multiclass classification
    if prediction_kind == "probability" and hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if family == "binary_classification" and proba.shape[1] == 2:
            col = output_columns[0] if output_columns else "prediction"
            return pd.DataFrame({col: proba[:, 1]})
        # Multiclass probabilities: emit matrix when output_columns length matches,
        # otherwise fall back to argmax label under the first output column.
        classes = getattr(pipeline, "classes_", None)
        if classes is None and hasattr(pipeline, "named_steps"):
            final = pipeline.named_steps.get("classifier")
            classes = getattr(final, "classes_", None)
        if output_columns and classes is not None and len(output_columns) == len(classes):
            return pd.DataFrame(proba, columns=output_columns)
        label = np.argmax(proba, axis=1)
        if classes is not None:
            label = np.asarray(classes)[label]
        col = output_columns[0] if output_columns else "prediction"
        return pd.DataFrame({col: label})

    preds = pipeline.predict(X_test)
    col = output_columns[0] if output_columns else "prediction"
    return pd.DataFrame({col: preds})


def _multi_label_positive_prob_matrix(raw) -> np.ndarray:
    """Normalize MultiOutputClassifier.predict_proba output → (n_samples, n_labels)."""
    if isinstance(raw, list):
        cols = []
        for arr in raw:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                cols.append(arr[:, 1])
            else:
                cols.append(arr.ravel())
        return np.column_stack(cols)
    arr = np.asarray(raw)
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, 1)


def build_submission(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    id_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
    submission_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble submission DataFrame per target_spec.submission contract.

    Priority:
        1. If submission_columns is provided, the result uses exactly those columns
           in that order (id_columns first then output_columns inferred).
        2. Else id_columns + output_columns.
        3. Else fallback: synthetic 'index' column + predictions columns.
    """
    id_columns = list(id_columns or [])
    output_columns = list(output_columns or list(predictions.columns))

    id_frame = _extract_id_frame(test_df, id_columns, len(predictions))

    prediction_frame = predictions.reset_index(drop=True).copy()
    # Make sure prediction columns include everything required.
    for col in output_columns:
        if col not in prediction_frame.columns:
            # Missing output column — fill with NaN so reviewer can catch it.
            prediction_frame[col] = np.nan
    prediction_frame = prediction_frame[output_columns]

    combined = pd.concat(
        [id_frame.reset_index(drop=True), prediction_frame], axis=1
    )

    if submission_columns:
        ordered = []
        for col in submission_columns:
            if col in combined.columns:
                ordered.append(col)
            else:
                # Column requested by schema but neither in id nor predictions.
                combined[col] = np.nan
                ordered.append(col)
        return combined[ordered]
    return combined


def _extract_id_frame(
    test_df: pd.DataFrame,
    id_columns: list[str],
    fallback_len: int,
) -> pd.DataFrame:
    if id_columns:
        missing = [c for c in id_columns if c not in test_df.columns]
        if not missing:
            return test_df[id_columns].reset_index(drop=True)
        # Partial availability: use what we have, synth the rest.
        available = [c for c in id_columns if c in test_df.columns]
        frame = test_df[available].reset_index(drop=True) if available else pd.DataFrame()
        for col in missing:
            frame[col] = range(fallback_len)
        return frame[id_columns]
    # Fallback: synthetic index
    return pd.DataFrame({"index": range(fallback_len)})


# ──────────────────────────────────────────────
# Results summary and persistence
# ──────────────────────────────────────────────

def summarize_cv_results(
    candidate_names: list[str],
    best_name: str,
    all_results: dict[str, dict[str, Any]],
    *,
    task_family: str = "",
    target_mode: str = "single_target",
    primary_metric: str = "",
) -> dict[str, Any]:
    """Build the model_results.json payload."""
    return {
        "candidate_models": list(candidate_names),
        "best_model_name": best_name,
        "best_params": all_results.get(best_name, {}).get("best_params", {}),
        "validation_metrics": {
            name: {k: v for k, v in r.items() if k != "best_params"}
            for name, r in all_results.items()
            if "error" not in r
        },
        "cv_results": all_results,
        "_meta": {
            "task_family": task_family,
            "target_mode": target_mode,
            "primary_metric": primary_metric,
        },
    }


def save_model_artifacts(
    best_pipeline,
    model_results_data: dict[str, Any],
    submission_df: pd.DataFrame | None,
    *,
    model_results_path: str = MODEL_RESULTS_PATH,
    best_model_path: str = BEST_MODEL_PATH,
    prediction_path: str = PREDICTION_PATH,
) -> dict[str, str]:
    """Persist model, CV report, and submission CSV to reserved artifact paths."""
    os.makedirs(Path(model_results_path).parent, exist_ok=True)
    os.makedirs(Path(best_model_path).parent, exist_ok=True)
    os.makedirs(Path(prediction_path).parent, exist_ok=True)

    prediction_file = ""
    if submission_df is not None and not submission_df.empty:
        submission_df.to_csv(prediction_path, index=False)
        prediction_file = prediction_path
    Path(model_results_path).write_text(
        json.dumps(model_results_data, indent=2, ensure_ascii=False, default=str)
    )
    if best_pipeline is not None:
        joblib.dump(best_pipeline, best_model_path)

    return {
        "model_results": model_results_path,
        "best_model": best_model_path if best_pipeline is not None else "",
        "prediction_file": prediction_file,
    }


# ──────────────────────────────────────────────
# Public re-exports for backward compatibility (existing tests)
# ──────────────────────────────────────────────

__all__ = [
    "MODEL_RESULTS_PATH",
    "BEST_MODEL_PATH",
    "PREDICTION_PATH",
    "normalize_task_family",
    "infer_task_family",
    "default_primary_metric",
    "metric_direction",
    "resolve_sklearn_scoring",
    "detect_feature_types",
    "build_preprocessor",
    "get_candidate_models",
    "optimize_candidate_search_space",
    "build_cv",
    "fit_and_select_model",
    "fit_and_select_clustering_model",
    "_resolve_candidate_parallelism",
    "predict_for_submission",
    "build_submission",
    "summarize_cv_results",
    "save_model_artifacts",
]
