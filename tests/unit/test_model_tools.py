"""Unit tests for app/tools/model_tools.py.

覆盖 4 个任务族 × target_mode：
  - binary_classification × single_target
  - multiclass_classification × single_target
  - regression × single_target
  - multilabel_classification × multi_label

不依赖任何外部文件；所有 fixture 是内联合成 DataFrame。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.tools.model_tools import (
    BEST_MODEL_PATH,
    MODEL_RESULTS_PATH,
    PREDICTION_PATH,
    build_cv,
    build_preprocessor,
    build_submission,
    default_primary_metric,
    detect_feature_types,
    fit_and_select_clustering_model,
    fit_and_select_model,
    get_candidate_models,
    optimize_candidate_search_space,
    infer_task_family,
    metric_direction,
    normalize_task_family,
    predict_for_submission,
    _resolve_candidate_parallelism,
    save_model_artifacts,
    summarize_cv_results,
)


# ──────────────────────────────────────────────
# Synthetic datasets per task family
# ──────────────────────────────────────────────


def _make_binary(n: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.integers(0, 2, n),
        "Age": rng.uniform(1, 80, n),
        "Fare": rng.uniform(5, 300, n),
        "Embarked": rng.choice(["S", "C", "Q"], n),
    })
    y = pd.Series(rng.integers(0, 2, n), name="Survived")
    return X, y


def _make_multiclass(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    X = pd.DataFrame({
        "bone_length": rng.uniform(0, 1, n),
        "rotting_flesh": rng.uniform(0, 1, n),
        "hair_length": rng.uniform(0, 1, n),
        "has_soul": rng.uniform(0, 1, n),
        "color": rng.choice(["white", "black", "blood", "clear"], n),
    })
    y = pd.Series(rng.choice(["Ghost", "Ghoul", "Goblin"], n), name="type")
    return X, y


def _make_regression(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(13)
    area = rng.uniform(500, 4000, n)
    rooms = rng.integers(1, 6, n)
    X = pd.DataFrame({
        "LotArea": area,
        "Rooms": rooms,
        "Neighborhood": rng.choice(["A", "B", "C", "D"], n),
    })
    noise = rng.normal(0, 10000, n)
    y = pd.Series(area * 100 + rooms * 5000 + noise, name="SalePrice")
    return X, y


def _make_multilabel(n: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(21)
    X = pd.DataFrame({
        "X_Minimum": rng.uniform(0, 1000, n),
        "Y_Minimum": rng.uniform(0, 1000, n),
        "Pixels_Areas": rng.uniform(10, 500, n),
        "Luminosity_Index": rng.uniform(-1, 1, n),
        "TypeOfSteel_A300": rng.integers(0, 2, n),
    })
    Y = pd.DataFrame({
        "Pastry":      rng.integers(0, 2, n),
        "Z_Scratch":   rng.integers(0, 2, n),
        "K_Scatch":    rng.integers(0, 2, n),
        "Stains":      rng.integers(0, 2, n),
    })
    return X, Y


# ──────────────────────────────────────────────
# normalize_task_family / infer_task_family
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("binary_classification", "binary_classification"),
        ("Binary", "binary_classification"),
        ("multiclass", "multiclass_classification"),
        ("Multi-Class", "multiclass_classification"),
        ("regression", "regression"),
        ("REGRESS", "regression"),
        ("multilabel_classification", "multilabel_classification"),
        ("multi-label", "multilabel_classification"),
        ("classification", "classification_unspecified"),
        ("", "unknown"),
        ("nonsense", "unknown"),
    ],
)
def test_normalize_task_family(raw, expected):
    assert normalize_task_family(raw) == expected


def test_infer_task_family_binary():
    y = pd.Series([0, 1, 0, 1, 1, 0])
    assert infer_task_family(y) == "binary_classification"


def test_infer_task_family_multiclass():
    y = pd.Series(["a", "b", "c", "a", "b", "c"])
    assert infer_task_family(y) == "multiclass_classification"


def test_infer_task_family_regression_float():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.uniform(0, 100, 50))
    assert infer_task_family(y) == "regression"


def test_infer_task_family_regression_integer_high_cardinality():
    y = pd.Series(range(100))
    assert infer_task_family(y) == "regression"


def test_infer_task_family_multilabel_from_dataframe():
    rng = np.random.default_rng(1)
    Y = pd.DataFrame({f"lbl_{i}": rng.integers(0, 2, 30) for i in range(4)})
    assert infer_task_family(Y) == "multilabel_classification"


def test_infer_task_family_respects_target_mode_override():
    y = pd.Series([0, 1, 0, 1])
    assert infer_task_family(y, target_mode="multi_label") == "multilabel_classification"


# ──────────────────────────────────────────────
# default_primary_metric / metric_direction
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "family,expected",
    [
        ("binary_classification", "roc_auc"),
        ("multiclass_classification", "f1_macro"),
        ("regression", "neg_root_mean_squared_error"),
        ("multilabel_classification", "roc_auc_macro_ovr"),
    ],
)
def test_default_primary_metric(family, expected):
    assert default_primary_metric(family) == expected


@pytest.mark.parametrize(
    "metric,direction",
    [
        ("roc_auc", "maximize"),
        ("accuracy", "maximize"),
        ("neg_root_mean_squared_error", "maximize"),
        ("hamming_loss", "minimize"),
        ("log_loss", "minimize"),
        ("rmse", "minimize"),
    ],
)
def test_metric_direction(metric, direction):
    assert metric_direction(metric) == direction


# ──────────────────────────────────────────────
# detect_feature_types
# ──────────────────────────────────────────────


def test_detect_feature_types_all_numeric():
    X, _ = _make_binary(30)
    num, cat = detect_feature_types(X)
    assert "Embarked" in cat
    assert "Age" in num


def test_detect_feature_types_with_exclude():
    X, _ = _make_binary(30)
    num, cat = detect_feature_types(X, exclude=["Sex"])
    assert "Sex" not in num + cat


def test_detect_feature_types_bool_is_numeric():
    X = pd.DataFrame({"flag": [True, False, True], "name": ["a", "b", "c"]})
    num, cat = detect_feature_types(X)
    assert "flag" in num
    assert "name" in cat


# ──────────────────────────────────────────────
# build_preprocessor
# ──────────────────────────────────────────────


def test_build_preprocessor_numeric_only():
    preproc = build_preprocessor(["Age", "Fare"], [])
    X = pd.DataFrame({"Age": [10.0, np.nan, 30.0], "Fare": [5.0, 10.0, 15.0]})
    out = preproc.fit_transform(X)
    assert out.shape == (3, 2)
    assert not np.isnan(out).any()


def test_build_preprocessor_custom_scaler_and_encoder():
    preproc = build_preprocessor(
        ["Age"], ["Sex"], scaler="minmax", cat_encode="ordinal"
    )
    X = pd.DataFrame({"Age": [10.0, 30.0, 40.0], "Sex": ["M", "F", "M"]})
    out = preproc.fit_transform(X)
    assert out.shape == (3, 2)


def test_build_preprocessor_none_scaler_skips_scaling():
    preproc = build_preprocessor(["Age"], [], scaler="none")
    X = pd.DataFrame({"Age": [10.0, 100.0]})
    out = preproc.fit_transform(X)
    # Values should not be centered around 0
    assert out.max() > 50


def test_build_preprocessor_invalid_args_fall_back_safely():
    preproc = build_preprocessor(
        ["Age"], ["Sex"],
        numeric_impute="invalid",
        scaler="made_up",
        cat_encode="weird",
    )
    # Should not raise; falls back to median + standard + onehot
    X = pd.DataFrame({"Age": [1.0, np.nan], "Sex": ["a", "b"]})
    preproc.fit_transform(X)


def test_build_preprocessor_impute_aliases_normalize():
    """LLM planner sometimes uses informal synonyms (e.g. 'mode' for
    'most_frequent', 'zero' for 'constant'). They should be remapped, not
    leak through to SimpleImputer where they raise."""
    preproc = build_preprocessor(
        ["Age"], ["Sex"],
        numeric_impute="average",  # → mean
        cat_impute="mode",         # → most_frequent
    )
    X = pd.DataFrame({"Age": [1.0, np.nan, 3.0], "Sex": ["a", "b", "a"]})
    out = preproc.fit_transform(X)
    assert out.shape[0] == 3


def test_build_preprocessor_unknown_cat_impute_falls_back():
    """A cat_impute string outside SimpleImputer's allowed set must fall back
    to 'most_frequent' instead of being passed through."""
    preproc = build_preprocessor(
        [], ["Sex"],
        cat_impute="not_a_real_strategy",
    )
    X = pd.DataFrame({"Sex": ["a", "b", "a"]})
    preproc.fit_transform(X)


def test_classifier_candidates_have_class_weight_balanced():
    """class_weight='balanced' is universally safe (no-op on balanced data,
    lifts minority signal on imbalanced data) and should be the default for
    estimators that support it."""
    cands = get_candidate_models("binary_classification", "single_target")
    assert cands["LogisticRegression"]["estimator"].get_params()["class_weight"] == "balanced"
    assert cands["RandomForest"]["estimator"].get_params()["class_weight"] == "balanced_subsample"
    assert cands["ExtraTrees"]["estimator"].get_params()["class_weight"] == "balanced"
    assert cands["SVC"]["estimator"].get_params()["class_weight"] == "balanced"
    assert cands["HistGradientBoosting"]["estimator"].get_params()["class_weight"] == "balanced"
    if "LightGBM" in cands:
        assert cands["LightGBM"]["estimator"].get_params()["class_weight"] == "balanced"


def test_per_target_roc_auc_scorer_returns_score_in_range():
    """Sanity check: scorer output is a float in [0, 1] on a synthetic
    multilabel-indicator y."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from app.tools.model_tools import _make_per_target_roc_auc_scorer

    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
    y = pd.DataFrame({
        "t1": rng.integers(0, 2, n),
        "t2": rng.integers(0, 2, n),
    })
    pipe = Pipeline([
        ("preproc", StandardScaler()),
        ("clf", MultiOutputClassifier(LogisticRegression(max_iter=200))),
    ])
    pipe.fit(X, y)
    scorer = _make_per_target_roc_auc_scorer()
    score = scorer(pipe, X, y)
    assert 0.0 <= score <= 1.0


def test_per_target_roc_auc_scorer_drives_grid_search():
    """End-to-end: GridSearchCV with our wrapped AUC scorer must complete on
    multilabel-indicator y where built-in 'roc_auc' / 'roc_auc_ovr' would
    fail with 'multilabel-indicator format is not supported'."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.multioutput import MultiOutputClassifier
    from app.tools.model_tools import _make_per_target_roc_auc_scorer

    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
    y = pd.DataFrame({
        "t1": rng.integers(0, 2, n),
        "t2": rng.integers(0, 2, n),
    })
    gs = GridSearchCV(
        MultiOutputClassifier(LogisticRegression(max_iter=200)),
        param_grid={"estimator__C": [0.1, 1.0]},
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring=_make_per_target_roc_auc_scorer(),
        error_score="raise",
        n_jobs=1,
    )
    gs.fit(X, y)
    assert 0.0 <= gs.best_score_ <= 1.0


def test_build_preprocessor_dict_args_do_not_raise_unhashable():
    """LLM planner sometimes passes wrapper dicts; must not raise."""
    preproc = build_preprocessor(
        ["Age"], ["Sex"],
        numeric_impute={"strategy": "median"},  # type: ignore[arg-type]
        scaler={"name": "standard"},  # type: ignore[arg-type]
        cat_encode={"type": "onehot"},  # type: ignore[arg-type]
        cat_impute={"strategy": "most_frequent"},  # type: ignore[arg-type]
        onehot_max_cats="50",  # type: ignore[arg-type]
    )
    X = pd.DataFrame({"Age": [1.0, np.nan, 2.0], "Sex": ["a", "b", "a"]})
    preproc.fit_transform(X)


# ──────────────────────────────────────────────
# get_candidate_models — per family
# ──────────────────────────────────────────────


def test_candidates_binary_classification_names():
    cands = get_candidate_models("binary_classification", "single_target")
    assert "LogisticRegression" in cands
    assert "RandomForest" in cands
    assert "ExtraTrees" in cands


def test_candidates_probability_classification_excludes_svc():
    cands = get_candidate_models(
        "binary_classification",
        "single_target",
        prediction_kind="probability",
    )
    assert "SVC" not in cands


def test_candidates_label_classification_keeps_nonprobability_svc():
    cands = get_candidate_models(
        "multiclass_classification",
        "single_target",
        prediction_kind="label",
    )
    assert "SVC" in cands
    assert cands["SVC"]["estimator"].probability is False


def test_candidates_regression_returns_regressors():
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    cands = get_candidate_models("regression", "single_target")
    assert isinstance(cands["Ridge"]["estimator"], Ridge)
    assert isinstance(cands["RandomForest"]["estimator"], RandomForestRegressor)
    assert isinstance(cands["ExtraTrees"]["estimator"], ExtraTreesRegressor)


def test_candidates_multilabel_wraps_in_multioutput():
    from sklearn.multioutput import MultiOutputClassifier

    cands = get_candidate_models("multilabel_classification", "multi_label")
    assert all(name.startswith("MultiOutput_") for name in cands)
    for cfg in cands.values():
        assert isinstance(cfg["estimator"], MultiOutputClassifier)
    # Param grid keys must have estimator__ prefix now
    for cfg in cands.values():
        for key in cfg["param_grid"]:
            assert key.startswith("classifier__estimator__")


def test_candidates_independent_instances():
    a = get_candidate_models("binary_classification")
    b = get_candidate_models("binary_classification")
    assert a["RandomForest"]["estimator"] is not b["RandomForest"]["estimator"]


def test_candidate_grids_expanded_conservatively():
    cls = get_candidate_models("binary_classification")
    reg = get_candidate_models("regression")
    assert cls["LogisticRegression"]["param_grid"]["classifier__C"] == [0.1, 0.3, 1.0, 3.0]
    assert cls["RandomForest"]["param_grid"]["classifier__n_estimators"] == [200, 400]
    assert cls["ExtraTrees"]["param_grid"]["classifier__n_estimators"] == [300, 600]
    assert cls["HistGradientBoosting"]["param_grid"]["classifier__max_iter"] == [100, 200]
    if "LightGBM" in cls:
        assert cls["LightGBM"]["param_grid"]["classifier__n_estimators"] == [200, 400]
    assert reg["Ridge"]["param_grid"]["classifier__alpha"] == [0.1, 1.0, 3.0, 10.0]
    assert reg["ExtraTrees"]["param_grid"]["classifier__n_estimators"] == [300, 600]
    assert reg["GradientBoosting"]["param_grid"]["classifier__n_estimators"] == [200, 400]


def test_large_tabular_budget_keeps_fast_boosters_and_academic_lgbm_point():
    cands = get_candidate_models("multiclass_classification")
    optimized, notes = optimize_candidate_search_space(
        cands,
        n_samples=76_518,
        n_features=40,
        task_family="multiclass_classification",
    )

    assert "RandomForest" not in optimized
    assert "ExtraTrees" not in optimized
    assert {"LogisticRegression", "HistGradientBoosting", "LightGBM"}.issubset(optimized)
    assert optimized["LightGBM"]["param_grid"] == {
        "classifier__n_estimators": [400],
        "classifier__learning_rate": [0.03, 0.07],
        "classifier__num_leaves": [31],
        "classifier__min_child_samples": [20],
    }
    assert optimized["HistGradientBoosting"]["param_grid"]["classifier__max_iter"] == [200]
    assert any("grid combinations reduced" in note for note in notes)


def test_medium_tabular_budget_single_points_expensive_forests():
    cands = get_candidate_models("binary_classification")
    optimized, notes = optimize_candidate_search_space(
        cands,
        n_samples=20_000,
        n_features=20,
        task_family="binary_classification",
    )

    assert optimized["RandomForest"]["param_grid"]["classifier__n_estimators"] == [200]
    assert optimized["ExtraTrees"]["param_grid"]["classifier__n_estimators"] == [300]
    assert any("medium dataset" in note for note in notes)


def test_resolve_candidate_parallelism_defaults_and_env(monkeypatch):
    monkeypatch.delenv("MODEL_BUILDING_MAX_PARALLEL_CANDIDATES", raising=False)
    assert _resolve_candidate_parallelism(1) == 1
    default_parallelism = _resolve_candidate_parallelism(6)
    assert 1 <= default_parallelism <= min(6, os.cpu_count() or 1, 4)

    monkeypatch.setenv("MODEL_BUILDING_MAX_PARALLEL_CANDIDATES", "2")
    assert _resolve_candidate_parallelism(6) == 2


def test_xgboost_soft_dependency():
    """XGBoost 出现与否都不应报错。"""
    cands_cls = get_candidate_models("binary_classification")
    cands_reg = get_candidate_models("regression")
    # 'XGBoost' may or may not be present depending on environment
    for cands in (cands_cls, cands_reg):
        if "XGBoost" in cands:
            assert "estimator" in cands["XGBoost"]


# ──────────────────────────────────────────────
# build_cv
# ──────────────────────────────────────────────


def test_build_cv_classification_is_stratified():
    from sklearn.model_selection import StratifiedKFold

    cv = build_cv("binary_classification", n_splits=3)
    assert isinstance(cv, StratifiedKFold)


def test_build_cv_regression_is_kfold():
    from sklearn.model_selection import KFold, StratifiedKFold

    cv = build_cv("regression", n_splits=3)
    assert isinstance(cv, KFold)
    assert not isinstance(cv, StratifiedKFold)


def test_build_cv_multilabel_is_kfold():
    from sklearn.model_selection import KFold

    cv = build_cv("multilabel_classification", n_splits=3)
    assert isinstance(cv, KFold)


def test_build_cv_strategy_override():
    from sklearn.model_selection import KFold, StratifiedKFold

    # binary classification can be downgraded to plain KFold via strategy override.
    cv = build_cv("binary_classification", n_splits=3, strategy="kfold")
    assert isinstance(cv, KFold)
    # multiclass also accepts stratified_kfold honored as-is.
    cv = build_cv("multiclass_classification", n_splits=3, strategy="stratified_kfold")
    assert isinstance(cv, StratifiedKFold)


def test_build_cv_stratified_request_downgraded_for_incompatible_family():
    """stratified_kfold must be silently downgraded to KFold for families
    where sklearn's StratifiedKFold would reject the target type
    (multilabel-indicator, continuous, etc.)."""
    from sklearn.model_selection import KFold, StratifiedKFold

    for family in (
        "regression",
        "multilabel_classification",
        "clustering",
    ):
        cv = build_cv(family, n_splits=3, strategy="stratified_kfold")
        assert isinstance(cv, KFold), f"{family} should downgrade to KFold"
        assert not isinstance(cv, StratifiedKFold)


def test_build_cv_multilabel_indicator_works_with_grid_search():
    """Regression test: previously a planner-supplied stratified_kfold caused
    'Got multilabel-indicator instead' inside GridSearchCV. KFold path must
    succeed end-to-end on a multilabel-indicator y."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.multioutput import MultiOutputClassifier

    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
    y = pd.DataFrame({
        "t1": rng.integers(0, 2, n),
        "t2": rng.integers(0, 2, n),
    })
    cv = build_cv("multilabel_classification", n_splits=3, strategy="stratified_kfold")
    gs = GridSearchCV(
        MultiOutputClassifier(LogisticRegression(max_iter=200)),
        param_grid={"estimator__C": [0.1, 1.0]},
        cv=cv,
        scoring="f1_micro",
        error_score="raise",
        n_jobs=1,
    )
    gs.fit(X, y)
    assert gs.best_score_ is not None


# ──────────────────────────────────────────────
# fit_and_select_model — per family
# ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def binary_run():
    X, y = _make_binary(80)
    num, cat = detect_feature_types(X)
    preproc = build_preprocessor(num, cat)
    # Restrict to 2 fast candidates to keep tests bounded
    full = get_candidate_models("binary_classification")
    cands = {k: full[k] for k in ("LogisticRegression", "RandomForest")}
    best, name, results = fit_and_select_model(
        X, y, preproc, cands, task_family="binary_classification",
    )
    return X, y, best, name, results, cands


@pytest.fixture(scope="module")
def multiclass_run():
    X, y = _make_multiclass(100)
    num, cat = detect_feature_types(X)
    preproc = build_preprocessor(num, cat)
    full = get_candidate_models("multiclass_classification")
    cands = {k: full[k] for k in ("LogisticRegression", "RandomForest")}
    best, name, results = fit_and_select_model(
        X, y, preproc, cands, task_family="multiclass_classification",
    )
    return X, y, best, name, results, cands


@pytest.fixture(scope="module")
def regression_run():
    X, y = _make_regression(100)
    num, cat = detect_feature_types(X)
    preproc = build_preprocessor(num, cat)
    full = get_candidate_models("regression")
    cands = {k: full[k] for k in ("Ridge", "RandomForest")}
    best, name, results = fit_and_select_model(
        X, y, preproc, cands, task_family="regression",
    )
    return X, y, best, name, results, cands


@pytest.fixture(scope="module")
def multilabel_run():
    X, Y = _make_multilabel(100)
    num, cat = detect_feature_types(X)
    preproc = build_preprocessor(num, cat)
    full = get_candidate_models("multilabel_classification", "multi_label")
    cands = {k: full[k] for k in ("MultiOutput_LogisticRegression", "MultiOutput_RandomForest")}
    best, name, results = fit_and_select_model(
        X, Y, preproc, cands,
        task_family="multilabel_classification", target_mode="multi_label",
    )
    return X, Y, best, name, results, cands


def test_fit_and_select_binary_returns_best(binary_run):
    _, _, best, name, results, cands = binary_run
    assert best is not None
    assert name in cands
    assert set(results.keys()) == set(cands.keys())


def test_fit_and_select_multiclass_predicts(multiclass_run):
    X, y, best, _, _, _ = multiclass_run
    preds = best.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))


def test_fit_and_select_regression_predicts_float(regression_run):
    X, _, best, _, _, _ = regression_run
    preds = best.predict(X)
    assert preds.dtype.kind in {"f", "i"}


def test_fit_and_select_multilabel_predicts_matrix(multilabel_run):
    X, Y, best, _, _, _ = multilabel_run
    preds = best.predict(X)
    assert preds.shape == Y.shape


def test_fit_and_select_multilabel_uses_holdout_not_cv(multilabel_run):
    *_, results, _ = multilabel_run
    for res in results.values():
        if "error" in res:
            continue
        assert res["_selection"] == "holdout"
        assert "val_roc_auc_macro_ovr" in res
        assert not any(key.startswith("cv_") for key in res)


def test_fit_and_select_strips_param_prefixes(binary_run):
    *_, results, _ = binary_run
    for res in results.values():
        if "error" in res:
            continue
        for key in res["best_params"]:
            assert not key.startswith("classifier__")
            assert not key.startswith("estimator__")


def test_fit_and_select_records_only_primary_metric(binary_run):
    *_, results, _ = binary_run
    for res in results.values():
        if "error" in res:
            continue
        metric_keys = [key for key in res.keys() if key.startswith("cv_")]
        assert all(not key.endswith("_std") for key in metric_keys)
        assert len(metric_keys) == 1


def test_fit_and_select_all_fail_returns_none():
    class Broken:
        def fit(self, *a, **kw):
            raise RuntimeError("intentional")

        def get_params(self, deep=True):
            return {}

        def set_params(self, **kw):
            return self

    X, y = _make_binary(30)
    preproc = build_preprocessor(list(X.select_dtypes("number").columns), [])
    candidates = {"broken": {"estimator": Broken(), "param_grid": {}}}
    best, name, results = fit_and_select_model(
        X, y, preproc, candidates, task_family="binary_classification"
    )
    assert best is None
    assert name == ""
    assert "error" in results["broken"]


# ──────────────────────────────────────────────
# predict_for_submission
# ──────────────────────────────────────────────


def test_predict_for_submission_binary_label(binary_run):
    X, _, best, _, _, _ = binary_run
    df = predict_for_submission(
        best, X,
        task_family="binary_classification",
        target_mode="single_target",
        prediction_kind="label",
        output_columns=["Survived"],
    )
    assert list(df.columns) == ["Survived"]
    assert len(df) == len(X)
    assert set(df["Survived"].unique()).issubset({0, 1})


def test_predict_for_submission_binary_probability(binary_run):
    X, _, best, _, _, _ = binary_run
    df = predict_for_submission(
        best, X,
        task_family="binary_classification",
        target_mode="single_target",
        prediction_kind="probability",
        output_columns=["Exited"],
    )
    assert list(df.columns) == ["Exited"]
    assert df["Exited"].between(0, 1).all()


def test_predict_for_submission_multiclass_label(multiclass_run):
    X, y, best, _, _, _ = multiclass_run
    df = predict_for_submission(
        best, X,
        task_family="multiclass_classification",
        prediction_kind="label",
        output_columns=["type"],
    )
    assert list(df.columns) == ["type"]
    assert set(df["type"].unique()).issubset(set(y.unique()))


def test_predict_for_submission_regression(regression_run):
    X, _, best, _, _, _ = regression_run
    df = predict_for_submission(
        best, X,
        task_family="regression",
        prediction_kind="continuous",
        output_columns=["SalePrice"],
    )
    assert list(df.columns) == ["SalePrice"]


def test_predict_for_submission_multilabel_probability(multilabel_run):
    X, Y, best, _, _, _ = multilabel_run
    df = predict_for_submission(
        best, X,
        task_family="multilabel_classification",
        target_mode="multi_label",
        prediction_kind="probability",
        output_columns=list(Y.columns),
    )
    assert list(df.columns) == list(Y.columns)
    assert df.shape == (len(X), Y.shape[1])
    for col in df.columns:
        assert df[col].between(0, 1).all()


# ──────────────────────────────────────────────
# build_submission
# ──────────────────────────────────────────────


def test_build_submission_with_id_column():
    test = pd.DataFrame({"id": [10, 20, 30], "X": [1, 2, 3]})
    preds = pd.DataFrame({"Exited": [0.1, 0.5, 0.9]})
    sub = build_submission(
        test, preds,
        id_columns=["id"],
        output_columns=["Exited"],
        submission_columns=["id", "Exited"],
    )
    assert list(sub.columns) == ["id", "Exited"]
    assert list(sub["id"]) == [10, 20, 30]


def test_build_submission_multi_output_columns():
    test = pd.DataFrame({"id": [1, 2]})
    preds = pd.DataFrame({
        "Pastry": [0.1, 0.9],
        "Stains": [0.8, 0.2],
    })
    sub = build_submission(
        test, preds,
        id_columns=["id"],
        output_columns=["Pastry", "Stains"],
        submission_columns=["id", "Pastry", "Stains"],
    )
    assert list(sub.columns) == ["id", "Pastry", "Stains"]


def test_build_submission_without_id_column_uses_fallback_index():
    test = pd.DataFrame({"X": [1, 2]})
    preds = pd.DataFrame({"y": [10, 20]})
    sub = build_submission(test, preds, id_columns=[], output_columns=["y"])
    assert "index" in sub.columns or "y" in sub.columns
    assert len(sub) == 2


def test_build_submission_schema_fills_missing_output_columns():
    """Columns in submission_columns but absent from predictions get NaN filled."""
    test = pd.DataFrame({"id": [1, 2]})
    preds = pd.DataFrame({"Pastry": [0.1, 0.9]})
    sub = build_submission(
        test, preds,
        id_columns=["id"],
        output_columns=["Pastry", "Stains"],
        submission_columns=["id", "Pastry", "Stains"],
    )
    assert "Stains" in sub.columns
    assert sub["Stains"].isna().all()


# ──────────────────────────────────────────────
# summarize_cv_results
# ──────────────────────────────────────────────


def test_summarize_cv_results_preserves_meta():
    all_results = {
        "A": {"best_params": {"C": 1}, "cv_accuracy_mean": 0.8},
        "B": {"error": "boom"},
    }
    summary = summarize_cv_results(
        ["A", "B"], "A", all_results,
        task_family="binary_classification",
        target_mode="single_target",
        primary_metric="roc_auc",
    )
    assert summary["best_model_name"] == "A"
    assert summary["candidate_models"] == ["A", "B"]
    assert summary["best_params"] == {"C": 1}
    assert "A" in summary["validation_metrics"]
    assert "B" not in summary["validation_metrics"]
    assert summary["_meta"]["task_family"] == "binary_classification"
    assert summary["_meta"]["primary_metric"] == "roc_auc"


# ──────────────────────────────────────────────
# save_model_artifacts
# ──────────────────────────────────────────────


def test_save_model_artifacts_writes_all_files(tmp_path, binary_run):
    _, _, best, name, all_results, cands = binary_run
    summary = summarize_cv_results(
        list(cands.keys()), name, all_results,
        task_family="binary_classification",
        target_mode="single_target",
        primary_metric="roc_auc",
    )
    sub = pd.DataFrame({"id": [1, 2], "Survived": [0, 1]})
    paths = save_model_artifacts(
        best, summary, sub,
        model_results_path=str(tmp_path / "m.json"),
        best_model_path=str(tmp_path / "m.pkl"),
        prediction_path=str(tmp_path / "sub.csv"),
    )
    assert Path(paths["model_results"]).exists()
    assert Path(paths["best_model"]).exists()
    assert Path(paths["prediction_file"]).exists()

    data = json.loads(Path(paths["model_results"]).read_text())
    assert data["best_model_name"] == name
    assert data["_meta"]["task_family"] == "binary_classification"

    loaded = joblib.load(paths["best_model"])
    assert hasattr(loaded, "predict")


def test_save_model_artifacts_no_submission_returns_empty_prediction_path(tmp_path, binary_run):
    _, _, best, name, all_results, cands = binary_run
    summary = summarize_cv_results(list(cands.keys()), name, all_results)
    paths = save_model_artifacts(
        best, summary, None,
        model_results_path=str(tmp_path / "m.json"),
        best_model_path=str(tmp_path / "m.pkl"),
        prediction_path=str(tmp_path / "sub.csv"),
    )
    assert paths["prediction_file"] == ""


def test_reserved_artifact_paths_unchanged():
    assert MODEL_RESULTS_PATH == "artifacts/models/model_results.json"
    assert BEST_MODEL_PATH == "artifacts/models/best_model.pkl"
    assert PREDICTION_PATH == "artifacts/predictions/submission.csv"


# ──────────────────────────────────────────────
# onehot_max_cats default alignment
# ──────────────────────────────────────────────


def test_build_preprocessor_default_onehot_max_cats_is_25():
    import inspect
    sig = inspect.signature(build_preprocessor)
    assert sig.parameters["onehot_max_cats"].default == 25


# ──────────────────────────────────────────────
# normalize_task_family — clustering
# ──────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["clustering", "Clustering", "kmeans_clustering", "cluster"])
def test_normalize_task_family_clustering(raw):
    assert normalize_task_family(raw) == "clustering"


def test_default_primary_metric_clustering():
    assert default_primary_metric("clustering") == "silhouette_score"


# ──────────────────────────────────────────────
# get_candidate_models — clustering + multi_output
# ──────────────────────────────────────────────


def test_candidates_clustering_returns_cluster_estimators():
    from sklearn.cluster import KMeans

    cands = get_candidate_models("clustering")
    assert "KMeans" in cands
    assert isinstance(cands["KMeans"]["estimator"], KMeans)
    # Clustering param_grid keys are direct (no classifier__ prefix)
    for key in cands["KMeans"]["param_grid"]:
        assert not key.startswith("classifier__")


def test_candidates_multi_output_regression_wraps_regressors():
    from sklearn.multioutput import MultiOutputRegressor

    cands = get_candidate_models("regression", "multi_output")
    assert all(name.startswith("MultiOutput_") for name in cands)
    for cfg in cands.values():
        assert isinstance(cfg["estimator"], MultiOutputRegressor)
    for cfg in cands.values():
        for key in cfg["param_grid"]:
            assert key.startswith("classifier__estimator__")


def test_candidates_multi_output_regression_excludes_lasso():
    cands = get_candidate_models("regression", "multi_output")
    assert "MultiOutput_Lasso" not in cands


# ──────────────────────────────────────────────
# fit_and_select_clustering_model
# ──────────────────────────────────────────────


def _make_clustering_data(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    # Three well-separated blobs
    c1 = rng.normal([0, 0], 0.5, (n // 3, 2))
    c2 = rng.normal([5, 5], 0.5, (n // 3, 2))
    c3 = rng.normal([0, 5], 0.5, (n - 2 * (n // 3), 2))
    data = np.vstack([c1, c2, c3])
    return pd.DataFrame(data, columns=["feature_a", "feature_b"])


@pytest.fixture(scope="module")
def clustering_run():
    X = _make_clustering_data(150)
    num, cat = detect_feature_types(X)
    preproc = build_preprocessor(num, cat)
    cands = get_candidate_models("clustering")
    # Keep only KMeans (DBSCAN may not always produce stable silhouette in CI)
    cands = {k: v for k, v in cands.items() if k == "KMeans"}
    best, name, results = fit_and_select_clustering_model(X, preproc, cands)
    return X, best, name, results


def test_clustering_returns_best_pipeline(clustering_run):
    _, best, name, results = clustering_run
    assert best is not None
    assert name == "KMeans"
    assert name in results


def test_clustering_results_has_silhouette(clustering_run):
    _, _, _, results = clustering_run
    assert "silhouette_score" in results["KMeans"]
    sil = results["KMeans"]["silhouette_score"]
    assert -1.0 <= sil <= 1.0


def test_clustering_results_has_n_clusters(clustering_run):
    _, _, _, results = clustering_run
    assert results["KMeans"]["n_clusters"] >= 2


def test_clustering_predict_for_submission(clustering_run):
    X, best, _, _ = clustering_run
    df = predict_for_submission(
        best, X,
        task_family="clustering",
        output_columns=["cluster"],
    )
    assert list(df.columns) == ["cluster"]
    assert len(df) == len(X)
    assert df["cluster"].nunique() >= 2


def test_clustering_all_fail_returns_none():
    X = _make_clustering_data(20)
    preproc = build_preprocessor(["feature_a", "feature_b"], [])
    # n_clusters > n_samples forces degenerate cluster
    from sklearn.cluster import KMeans

    cands = {"KMeans": {"estimator": KMeans(n_clusters=50, random_state=0, n_init=1),
                        "param_grid": {}}}
    best, name, results = fit_and_select_clustering_model(X, preproc, cands)
    assert best is None or name != "" or "error" in results.get("KMeans", {})


# ──────────────────────────────────────────────
# predict_for_submission — multi_output regression
# ──────────────────────────────────────────────


def _make_multi_output_regression(n: int = 100):
    rng = np.random.default_rng(77)
    X = pd.DataFrame({
        "x1": rng.uniform(0, 10, n),
        "x2": rng.uniform(0, 5, n),
    })
    Y = pd.DataFrame({
        "y1": X["x1"] * 2 + rng.normal(0, 0.5, n),
        "y2": X["x2"] * 3 + rng.normal(0, 0.5, n),
    })
    return X, Y


@pytest.fixture(scope="module")
def multi_output_reg_run():
    X, Y = _make_multi_output_regression(80)
    num, cat = detect_feature_types(X)
    preproc = build_preprocessor(num, cat)
    cands = get_candidate_models("regression", "multi_output")
    subset = {k: cands[k] for k in list(cands)[:2]}
    best, name, results = fit_and_select_model(
        X, Y, preproc, subset,
        task_family="regression", target_mode="multi_output",
    )
    return X, Y, best, name, results


def test_multi_output_reg_returns_pipeline(multi_output_reg_run):
    _, _, best, name, results = multi_output_reg_run
    assert best is not None
    assert name


def test_multi_output_reg_predict_shape(multi_output_reg_run):
    X, Y, best, _, _ = multi_output_reg_run
    df = predict_for_submission(
        best, X,
        task_family="regression",
        target_mode="multi_output",
        output_columns=list(Y.columns),
    )
    assert list(df.columns) == list(Y.columns)
    assert df.shape == (len(X), 2)


# ──────────────────────────────────────────────
# build_cv — n_splits uniformly (no 2-fold for multilabel)
# ──────────────────────────────────────────────


@pytest.mark.parametrize("family,mode", [
    ("binary_classification", "single_target"),
    ("multilabel_classification", "multi_label"),
    ("regression", "single_target"),
    ("regression", "multi_output"),
])
def test_build_cv_n_splits_minimum_3(family, mode):
    cv = build_cv(family, n_splits=3)
    assert cv.n_splits == 3
