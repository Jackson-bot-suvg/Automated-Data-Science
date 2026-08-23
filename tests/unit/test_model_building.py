"""End-to-end unit tests for app/agents/model_building.run_model_building.

覆盖 4 个任务族 × target_mode：
  - binary_classification / single_target
  - multiclass_classification / single_target
  - regression / single_target
  - multilabel_classification / multi_label

所有 fixture 是内联合成 DataFrame；LLM Planner 失败路径走默认 config
（仍能产出完整 ModelState / artifacts），所以我们不 mock LLM——直接让
``BasePlanner.plan_model_building`` 在测试环境里抛错即可（默认 plan fallback）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents.model_building import run_model_building
from app.schemas.state_schema import (
    EvaluationSpec,
    FeatureState,
    GraphState,
    ModelState,
    SubmissionSpec,
    TargetSpec,
)


# ──────────────────────────────────────────────
# Synthetic CSV fixtures
# ──────────────────────────────────────────────


def _binary_train(n=80):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id": np.arange(n),
        "Survived": rng.integers(0, 2, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.integers(0, 2, n),
        "Age": rng.uniform(1, 80, n),
        "Fare": rng.uniform(5, 300, n),
        "Embarked": rng.choice(["S", "C", "Q"], n),
    })


def _binary_test(n=30):
    return _binary_train(n).drop(columns=["Survived"])


def _multiclass_train(n=120):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "id": np.arange(n),
        "type": rng.choice(["Ghost", "Ghoul", "Goblin"], n),
        "bone_length": rng.uniform(0, 1, n),
        "rotting_flesh": rng.uniform(0, 1, n),
        "hair_length": rng.uniform(0, 1, n),
        "has_soul": rng.uniform(0, 1, n),
        "color": rng.choice(["white", "black", "blood"], n),
    })


def _multiclass_test(n=30):
    return _multiclass_train(n).drop(columns=["type"])


def _regression_train(n=100):
    rng = np.random.default_rng(13)
    area = rng.uniform(500, 4000, n)
    rooms = rng.integers(1, 6, n)
    return pd.DataFrame({
        "Id": np.arange(n),
        "LotArea": area,
        "Rooms": rooms,
        "Neighborhood": rng.choice(["A", "B", "C"], n),
        "SalePrice": area * 100 + rooms * 5000 + rng.normal(0, 5000, n),
    })


def _regression_test(n=25):
    return _regression_train(n).drop(columns=["SalePrice"])


def _multilabel_train(n=120, target_columns=None):
    rng = np.random.default_rng(21)
    target_columns = target_columns or ["Pastry", "Z_Scratch", "K_Scatch", "Stains"]
    base = pd.DataFrame({
        "id": np.arange(n),
        "X_Minimum": rng.uniform(0, 1000, n),
        "Y_Minimum": rng.uniform(0, 1000, n),
        "Pixels_Areas": rng.uniform(10, 500, n),
        "Luminosity_Index": rng.uniform(-1, 1, n),
        "TypeOfSteel_A300": rng.integers(0, 2, n),
    })
    for col in target_columns:
        base[col] = rng.integers(0, 2, n)
    return base


def _multilabel_test(n=30, target_columns=None):
    target_columns = target_columns or ["Pastry", "Z_Scratch", "K_Scatch", "Stains"]
    return _multilabel_train(n, target_columns).drop(columns=target_columns)


def _multitarget_classification_train(n=120):
    """Multi-column targets where at least one column has 3+ classes.

    Modeling scope assumes targets are already numerically encoded — string
    label encoding is upstream (reader / feature engineering) territory.
    """
    rng = np.random.default_rng(31)
    return pd.DataFrame({
        "id": np.arange(n),
        "feat_a": rng.uniform(0, 1, n),
        "feat_b": rng.uniform(0, 1, n),
        "feat_c": rng.choice(["x", "y", "z"], n),
        # binary target
        "is_defect": rng.integers(0, 2, n),
        # 3-class target → triggers multiclass-multioutput subtype
        "severity": rng.integers(0, 3, n),
    })


def _multitarget_classification_test(n=25):
    df = _multitarget_classification_train(n)
    return df.drop(columns=["is_defect", "severity"])


# ──────────────────────────────────────────────
# State builder helper
# ──────────────────────────────────────────────


def _state_for_task(
    tmp_path,
    *,
    train_df,
    test_df,
    target_columns,
    task_family,
    target_mode="single_target",
    prediction_kind="label",
    id_columns=(),
    output_columns=(),
    submission_columns=(),
    primary_metric="",
) -> GraphState:
    train_path = tmp_path / "engineered_train.csv"
    test_path = tmp_path / "engineered_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    submission = SubmissionSpec(
        id_columns=list(id_columns),
        output_columns=list(output_columns) or list(target_columns),
        submission_columns=list(submission_columns) or (list(id_columns) + list(target_columns)),
    )
    target_spec = TargetSpec(
        status="resolved",
        target_columns=list(target_columns),
        target_mode=target_mode,
        task_family=task_family,
        prediction_kind=prediction_kind,
        submission=submission,
        evaluation=EvaluationSpec(primary_metric=primary_metric or ""),
    )
    selected_features = [
        c for c in train_df.columns
        if c not in list(target_columns) + list(id_columns)
    ]
    return GraphState(
        target_spec=target_spec,
        feature_state=FeatureState(
            engineered_train_path=str(train_path),
            engineered_test_path=str(test_path),
            selected_features=selected_features,
        ),
    )


def _run(state: GraphState):
    return asyncio.get_event_loop().run_until_complete(run_model_building(state))


# ──────────────────────────────────────────────
# Path validation & error envelope
# ──────────────────────────────────────────────


def test_missing_train_path_returns_error():
    state = GraphState(
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["Survived"],
            task_family="binary_classification",
            target_mode="single_target",
        ),
        feature_state=FeatureState(
            engineered_train_path="nonexistent/train.csv",
            engineered_test_path="nonexistent/test.csv",
        ),
    )
    result = _run(state)
    assert result["runtime_state"]["interrupted"] is True
    assert "review_state" not in result  # retry_count_by_phase must not be reset


def test_target_not_in_train_returns_error(tmp_path):
    train = _binary_train().drop(columns=["Survived"])
    test = _binary_test()
    state = _state_for_task(
        tmp_path,
        train_df=train, test_df=test,
        target_columns=["Survived"],
        task_family="binary_classification",
    )
    result = _run(state)
    assert result["runtime_state"]["interrupted"] is True
    assert "review_state" not in result  # retry_count_by_phase must not be reset


def test_all_candidates_failure_records_each_cause(tmp_path, monkeypatch):
    """When every candidate raises, each cause must surface in data_quality_flags."""

    def _fake_fit_and_select_model(*args, **kwargs):
        return (
            None,
            "",
            {
                "LogisticRegression": {"error": "convergence failure on synthetic data"},
                "RandomForest": {"error": "n_jobs deadlock"},
            },
        )

    import app.agents.model_building as mb

    monkeypatch.setattr(mb, "fit_and_select_model", _fake_fit_and_select_model)

    state = _state_for_task(
        tmp_path,
        train_df=_binary_train(), test_df=_binary_test(),
        target_columns=["Survived"],
        task_family="binary_classification",
        target_mode="single_target",
        id_columns=["id"],
    )
    result = _run(state)

    assert result["runtime_state"]["interrupted"] is True
    assert "review_state" not in result  # do not reset retry_count_by_phase
    flags = result["data_state"]["data_quality_flags"]
    joined = " | ".join(flags)
    assert "LogisticRegression" in joined
    assert "convergence failure" in joined
    assert "RandomForest" in joined
    assert "n_jobs deadlock" in joined


# ──────────────────────────────────────────────
# Binary classification (Titanic-like)
# ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def binary_result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("binary")
    state = _state_for_task(
        tmp,
        train_df=_binary_train(), test_df=_binary_test(),
        target_columns=["Survived"],
        task_family="binary_classification",
        target_mode="single_target",
        prediction_kind="label",
        id_columns=["id"],
        submission_columns=["id", "Survived"],
    )
    return _run(state)


def test_binary_state_fields(binary_result):
    ms = binary_result["model_state"]
    assert ms["best_model_name"]
    assert ms["candidate_models"]
    assert ms["best_model_name"] in ms["candidate_models"]
    assert ms["validation_metrics"]["_meta"]["task_family"] == "binary_classification"
    assert ms["validation_metrics"]["_meta"]["target_mode"] == "single_target"


def test_binary_artifacts_written(binary_result):
    arts = binary_result["artifacts"]
    assert Path(arts["model_results"]).exists()
    assert Path(arts["best_model"]).exists()
    assert Path(arts["prediction_file"]).exists()
    sub = pd.read_csv(arts["prediction_file"])
    assert list(sub.columns) == ["id", "Survived"]


def test_binary_baseline_metrics_shape(binary_result):
    metrics = binary_result["model_state"]["baseline_metrics"]
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["_meta"]["source"] == "out_of_fold_dummy_baseline"


# ──────────────────────────────────────────────
# Binary classification with probability output (bank_churn-style)
# ──────────────────────────────────────────────


def test_binary_probability_submission(tmp_path):
    state = _state_for_task(
        tmp_path,
        train_df=_binary_train(), test_df=_binary_test(),
        target_columns=["Survived"],
        task_family="binary_classification",
        target_mode="single_target",
        prediction_kind="probability",
        id_columns=["id"],
        output_columns=["Exited"],            # rename for clarity
        submission_columns=["id", "Exited"],
    )
    result = _run(state)
    sub = pd.read_csv(result["artifacts"]["prediction_file"])
    assert list(sub.columns) == ["id", "Exited"]
    # Probability outputs must be in [0,1]
    assert sub["Exited"].between(0, 1).all()


# ──────────────────────────────────────────────
# Multiclass classification
# ──────────────────────────────────────────────


def test_multiclass_classification(tmp_path):
    state = _state_for_task(
        tmp_path,
        train_df=_multiclass_train(), test_df=_multiclass_test(),
        target_columns=["type"],
        task_family="multiclass_classification",
        target_mode="single_target",
        prediction_kind="label",
        id_columns=["id"],
        submission_columns=["id", "type"],
    )
    result = _run(state)
    ms = result["model_state"]
    assert ms["validation_metrics"]["_meta"]["task_family"] == "multiclass_classification"
    sub = pd.read_csv(result["artifacts"]["prediction_file"])
    assert list(sub.columns) == ["id", "type"]
    # Predictions must be one of the known labels
    assert set(sub["type"].unique()).issubset({"Ghost", "Ghoul", "Goblin"})
    metrics = ms["baseline_metrics"]
    assert "f1_macro" in metrics


# ──────────────────────────────────────────────
# Regression
# ──────────────────────────────────────────────


def test_regression(tmp_path):
    state = _state_for_task(
        tmp_path,
        train_df=_regression_train(), test_df=_regression_test(),
        target_columns=["SalePrice"],
        task_family="regression",
        target_mode="single_target",
        prediction_kind="continuous",
        id_columns=["Id"],
        submission_columns=["Id", "SalePrice"],
    )
    result = _run(state)
    ms = result["model_state"]
    assert ms["validation_metrics"]["_meta"]["task_family"] == "regression"
    sub = pd.read_csv(result["artifacts"]["prediction_file"])
    assert list(sub.columns) == ["Id", "SalePrice"]
    # Continuous predictions
    assert sub["SalePrice"].dtype.kind in {"f", "i"}
    metrics = ms["baseline_metrics"]
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["_meta"]["source"] == "out_of_fold_dummy_baseline"


def test_regression_metric_aliases_from_reader_are_sanitized(tmp_path):
    state = _state_for_task(
        tmp_path,
        train_df=_regression_train(), test_df=_regression_test(),
        target_columns=["SalePrice"],
        task_family="regression",
        target_mode="single_target",
        prediction_kind="continuous",
        id_columns=["Id"],
        submission_columns=["Id", "SalePrice"],
        primary_metric="rmse",
    )
    state.target_spec.evaluation.candidate_metrics = ["rmse", "mae", "r2"]
    result = _run(state)
    meta = result["model_state"]["validation_metrics"]["_meta"]
    assert meta["primary_metric"] == "neg_root_mean_squared_error"


# ──────────────────────────────────────────────
# Multi-label classification (plate_defect-like)
# ──────────────────────────────────────────────


def test_multilabel_uses_per_target_auc_scorer(tmp_path):
    """When primary_metric is roc_auc_macro_ovr (the upstream-supplied default
    for multilabel), GridSearchCV must actually optimize per-target AUC instead
    of silently degrading to f1_micro. cv_results should expose the AUC key."""
    targets = ["Pastry", "Z_Scratch", "K_Scatch", "Stains"]
    state = _state_for_task(
        tmp_path,
        train_df=_multilabel_train(target_columns=targets),
        test_df=_multilabel_test(target_columns=targets),
        target_columns=targets,
        task_family="multilabel_classification",
        target_mode="multi_label",
        prediction_kind="probability",
        id_columns=["id"],
        output_columns=targets,
        submission_columns=["id"] + targets,
        primary_metric="roc_auc_macro_ovr",
    )
    result = _run(state)
    ms = result["model_state"]
    cv = ms["cv_results"][ms["best_model_name"]]
    cv_keys = list(cv.keys())
    assert any(k.startswith("cv_per_target_auc_") for k in cv_keys), (
        f"expected per_target_auc in cv_results, got {cv_keys}"
    )


def test_multilabel_classification(tmp_path):
    targets = ["Pastry", "Z_Scratch", "K_Scatch", "Stains"]
    state = _state_for_task(
        tmp_path,
        train_df=_multilabel_train(target_columns=targets),
        test_df=_multilabel_test(target_columns=targets),
        target_columns=targets,
        task_family="multilabel_classification",
        target_mode="multi_label",
        prediction_kind="probability",
        id_columns=["id"],
        output_columns=targets,
        submission_columns=["id"] + targets,
    )
    result = _run(state)
    ms = result["model_state"]
    assert ms["validation_metrics"]["_meta"]["task_family"] == "multilabel_classification"
    assert ms["validation_metrics"]["_meta"]["target_mode"] == "multi_label"
    # best_model_name remains a single string (MultiOutput_<Name>)
    assert isinstance(ms["best_model_name"], str)
    assert ms["best_model_name"].startswith("MultiOutput_")
    sub = pd.read_csv(result["artifacts"]["prediction_file"])
    assert list(sub.columns) == ["id"] + targets
    # Probability outputs must be in [0,1]
    for col in targets:
        assert sub[col].between(0, 1).all()
    metrics = ms["baseline_metrics"]
    assert "hamming_loss" in metrics
    assert "f1_macro" in metrics
    assert metrics["_meta"]["source"] == "out_of_fold_dummy_baseline"
    assert result["feature_state"]["feature_importance_summary"]
    assert result["model_state"]["model_explanations"]


def test_multitarget_classification_with_multiclass_column(tmp_path):
    """Targets are not all binary (one column has 3 classes) — modeling must
    detect multiclass-multioutput subtype, route to subset-accuracy scoring,
    and write a label submission instead of probabilities."""
    targets = ["is_defect", "severity"]
    state = _state_for_task(
        tmp_path,
        train_df=_multitarget_classification_train(),
        test_df=_multitarget_classification_test(),
        target_columns=targets,
        task_family="multilabel_classification",
        target_mode="multi_label",
        prediction_kind="probability",  # should auto-fallback to label
        id_columns=["id"],
        output_columns=targets,
        submission_columns=["id"] + targets,
    )
    result = _run(state)
    ms = result["model_state"]
    # Validation metrics must include the per-target macro F1 we wired in.
    val = ms["validation_metrics"]
    assert val["_meta"]["task_family"] == "multilabel_classification"
    # at least one cv key contains accuracy or per_target_macro_f1 means
    cv_keys = [
        k for k in ms["cv_results"][ms["best_model_name"]]
        if k.startswith("cv_") and k.endswith("_mean")
    ]
    assert any("accuracy" in k or "per_target_macro_f1" in k for k in cv_keys), (
        f"expected subset-accuracy or per-target-macro-F1 in CV keys: {cv_keys}"
    )
    # Submission contains hard labels — severity values come from the original
    # category set, not floats in [0,1].
    sub = pd.read_csv(result["artifacts"]["prediction_file"])
    assert list(sub.columns) == ["id"] + targets
    # severity is encoded as int [0,1,2]; predictions stay within that label set.
    severity_unique = set(sub["severity"].unique().tolist())
    assert severity_unique.issubset({0, 1, 2}), severity_unique
    # is_defect remains binary 0/1.
    assert set(sub["is_defect"].unique().tolist()).issubset({0, 1})
    # Baseline is intentionally skipped for multiclass_multioutput.
    assert ms["baseline_metrics"] == {}


def test_multilabel_legacy_scope_is_normalized(tmp_path):
    targets = ["Pastry", "Z_Scratch", "K_Scatch", "Stains"]
    state = _state_for_task(
        tmp_path,
        train_df=_multilabel_train(target_columns=targets),
        test_df=_multilabel_test(target_columns=targets),
        target_columns=targets,
        task_family="classification",
        target_mode="multi",
        prediction_kind="probability",
        id_columns=["id"],
        output_columns=targets,
        submission_columns=["id"] + targets,
        primary_metric="f1",
    )
    state.target_spec.evaluation.candidate_metrics = ["f1", "roc_auc"]
    result = _run(state)
    meta = result["model_state"]["validation_metrics"]["_meta"]
    assert meta["task_family"] == "multilabel_classification"
    assert meta["target_mode"] == "multi_label"
    assert meta["primary_metric"] in {"f1_micro", "roc_auc_macro_ovr"}


# ──────────────────────────────────────────────
# Fallback: target_spec empty, legacy mirrors present
# ──────────────────────────────────────────────


def test_target_spec_empty_uses_legacy_mirror(tmp_path):
    """When target_spec is empty, modeling falls back to ProblemUnderstanding mirror."""
    from app.schemas.state_schema import ProblemUnderstanding

    train = _binary_train()
    test = _binary_test()
    train_path = tmp_path / "tr.csv"
    test_path = tmp_path / "te.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    state = GraphState(
        # target_spec intentionally empty
        problem_understanding=ProblemUnderstanding(
            target_column="Survived",
            target_columns=["Survived"],
            predicted_task_type="binary_classification",
        ),
        feature_state=FeatureState(
            engineered_train_path=str(train_path),
            engineered_test_path=str(test_path),
        ),
    )
    result = _run(state)
    assert "runtime_state" not in result or not result.get("runtime_state", {}).get("interrupted")
    ms = result["model_state"]
    assert ms["best_model_name"]
    # data_quality_flags should record the fallback warning
    flags = result["data_state"]["data_quality_flags"]
    assert any("fell back to legacy mirror" in f for f in flags)


def test_task_family_inference_from_y(tmp_path):
    """When task_family is empty/unknown, infer from y data."""
    train = _regression_train()
    test = _regression_test()
    train_path = tmp_path / "tr.csv"
    test_path = tmp_path / "te.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    state = GraphState(
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["SalePrice"],
            target_mode="single_target",
            task_family="",                   # deliberately empty
            submission=SubmissionSpec(id_columns=["Id"], output_columns=["SalePrice"]),
        ),
        feature_state=FeatureState(
            engineered_train_path=str(train_path),
            engineered_test_path=str(test_path),
        ),
    )
    result = _run(state)
    meta = result["model_state"]["validation_metrics"]["_meta"]
    assert meta["task_family"] == "regression"
    flags = result["data_state"]["data_quality_flags"]
    assert any("task_family inferred" in f for f in flags)


# ──────────────────────────────────────────────
# Return shape invariants
# ──────────────────────────────────────────────


def test_return_keys(binary_result):
    # Allow extra keys (data_state) for the flag propagation
    assert {"model_state", "artifacts", "current_phase", "phase_history"}.issubset(
        binary_result.keys()
    )


def test_current_phase(binary_result):
    assert binary_result["current_phase"] == "model_building"


def test_model_state_is_mergeable(binary_result):
    ms = binary_result["model_state"]
    # ModelState schema should accept this dict (minus _meta which is nested in validation_metrics)
    validation_metrics = {k: v for k, v in ms["validation_metrics"].items() if k != "_meta"}
    ms_for_schema = {**ms, "validation_metrics": validation_metrics}
    ModelState(**{k: v for k, v in ms_for_schema.items() if k in ModelState.model_fields})


def test_model_results_meta_present(binary_result):
    meta = binary_result["model_state"]["validation_metrics"]["_meta"]
    assert meta["task_family"] == "binary_classification"
    assert meta["primary_metric"]
    assert meta["metric_direction"] in {"maximize", "minimize"}


# ──────────────────────────────────────────────
# NaN in test set handling (SimpleImputer fallback)
# ──────────────────────────────────────────────


def test_nan_in_test_set_handled(tmp_path):
    train = _binary_train()
    test = _binary_test()
    test.loc[0, "Fare"] = float("nan")
    state = _state_for_task(
        tmp_path,
        train_df=train, test_df=test,
        target_columns=["Survived"],
        task_family="binary_classification",
        id_columns=["id"],
        submission_columns=["id", "Survived"],
    )
    result = _run(state)
    assert "runtime_state" not in result or not result.get("runtime_state", {}).get("interrupted")


def test_all_missing_feature_columns_dropped(tmp_path):
    """Columns that are entirely NaN in training data are dropped without error."""
    train = _binary_train()
    train["all_nan_col"] = float("nan")
    test = _binary_test()
    test["all_nan_col"] = float("nan")
    state = _state_for_task(
        tmp_path,
        train_df=train, test_df=test,
        target_columns=["Survived"],
        task_family="binary_classification",
        id_columns=["id"],
        submission_columns=["id", "Survived"],
    )
    result = _run(state)
    assert "runtime_state" not in result or not result.get("runtime_state", {}).get("interrupted")
    flags = result["data_state"]["data_quality_flags"]
    assert any("all_nan_col" in f for f in flags)


# ──────────────────────────────────────────────
# CV fold count: multilabel must use 3-fold (not 2)
# ──────────────────────────────────────────────


def test_multi_output_non_regression_is_downgraded(tmp_path):
    """multi_output target_mode with classification family must be downgraded to single_target."""
    train = _binary_train()
    test = _binary_test()
    state = _state_for_task(
        tmp_path,
        train_df=train, test_df=test,
        target_columns=["Survived"],
        task_family="binary_classification",
        target_mode="multi_output",   # nonsensical but should be handled
        id_columns=["id"],
        submission_columns=["id", "Survived"],
    )
    result = _run(state)
    # Must not crash; target_mode is downgraded to single_target internally
    assert "runtime_state" not in result or not result.get("runtime_state", {}).get("interrupted")
    meta = result["model_state"]["validation_metrics"]["_meta"]
    assert meta["target_mode"] == "single_target"


def test_target_mode_inferred_as_multi_output_for_numeric_multi_col(tmp_path):
    """When target_mode=unknown and targets are numeric multi-column, infer multi_output."""
    train = _multi_output_reg_train()
    test = _multi_output_reg_test()
    from app.schemas.state_schema import EvaluationSpec, FeatureState, SubmissionSpec, TargetSpec

    train_path = tmp_path / "tr.csv"
    test_path = tmp_path / "te.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    state = GraphState(
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["y1", "y2"],
            target_mode="unknown",       # let the agent infer
            task_family="regression",
            submission=SubmissionSpec(output_columns=["y1", "y2"]),
        ),
        feature_state=FeatureState(
            engineered_train_path=str(train_path),
            engineered_test_path=str(test_path),
            selected_features=["feat1", "feat2", "cat"],
        ),
    )
    result = _run(state)
    assert "runtime_state" not in result or not result.get("runtime_state", {}).get("interrupted")
    flags = result["data_state"]["data_quality_flags"]
    assert any("multi_output" in f for f in flags)


def test_multilabel_uses_3fold_cv(tmp_path):
    """Regression guard: _default_cv must not set n_splits=2 for multilabel."""
    from app.agents.model_building import _default_cv, ResolvedTargetScope

    scope = ResolvedTargetScope(
        target_columns=["A", "B"],
        task_family="multilabel_classification",
        target_mode="multi_label",
        prediction_kind="label",
        primary_metric="roc_auc_macro_ovr",
        candidate_metrics=[],
        id_columns=[],
        output_columns=["A", "B"],
        submission_columns=["A", "B"],
    )
    cv_cfg = _default_cv(scope)
    assert cv_cfg["n_splits"] == 3, (
        f"multilabel CV must use 3-fold, got {cv_cfg['n_splits']}"
    )


# ──────────────────────────────────────────────
# Multi-output regression (end-to-end)
# ──────────────────────────────────────────────


def _multi_output_reg_train(n=100):
    rng = np.random.default_rng(77)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    return pd.DataFrame({
        "feat1": x1,
        "feat2": x2,
        "cat": rng.choice(["A", "B", "C"], n),
        "y1": x1 * 2.0 + rng.normal(0, 0.3, n),
        "y2": x2 * 3.0 + rng.normal(0, 0.3, n),
    })


def _multi_output_reg_test(n=25):
    return _multi_output_reg_train(n).drop(columns=["y1", "y2"])


def test_multi_output_regression(tmp_path):
    from app.schemas.state_schema import EvaluationSpec, FeatureState, SubmissionSpec, TargetSpec

    train = _multi_output_reg_train()
    test = _multi_output_reg_test()
    train_path = tmp_path / "tr.csv"
    test_path = tmp_path / "te.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    state = GraphState(
        target_spec=TargetSpec(
            status="resolved",
            target_columns=["y1", "y2"],
            target_mode="multi_output",
            task_family="regression",
            prediction_kind="continuous",
            submission=SubmissionSpec(
                output_columns=["y1", "y2"],
                submission_columns=["y1", "y2"],
            ),
            evaluation=EvaluationSpec(primary_metric="neg_root_mean_squared_error"),
        ),
        feature_state=FeatureState(
            engineered_train_path=str(train_path),
            engineered_test_path=str(test_path),
            selected_features=["feat1", "feat2", "cat"],
        ),
    )
    result = _run(state)
    assert "runtime_state" not in result or not result.get("runtime_state", {}).get("interrupted")
    ms = result["model_state"]
    assert ms["best_model_name"].startswith("MultiOutput_")
    assert ms["validation_metrics"]["_meta"]["task_family"] == "regression"
    assert ms["validation_metrics"]["_meta"]["target_mode"] == "multi_output"
    sub = pd.read_csv(result["artifacts"]["prediction_file"])
    assert "y1" in sub.columns and "y2" in sub.columns
    assert sub.shape[1] == 2
    assert sub.shape[0] == len(test)


# ──────────────────────────────────────────────
# Clustering (end-to-end, no target columns)
# ──────────────────────────────────────────────


def _clustering_train(n=150):
    rng = np.random.default_rng(99)
    c1 = rng.normal([0, 0], 0.4, (n // 3, 2))
    c2 = rng.normal([6, 6], 0.4, (n // 3, 2))
    c3 = rng.normal([0, 6], 0.4, (n - 2 * (n // 3), 2))
    data = np.vstack([c1, c2, c3])
    return pd.DataFrame(data, columns=["feat_a", "feat_b"])


# ──────────────────────────────────────────────
# Unit tests for internal helpers
# ──────────────────────────────────────────────


def test_apply_plan_to_config_parses_steps():
    """_apply_plan_to_config merges plan steps into ModelingConfig."""
    from app.agents.model_building import ModelingConfig, ResolvedTargetScope, _apply_plan_to_config

    class FakeStep:
        def __init__(self, tool, parameters):
            self.tool = tool
            self.parameters = parameters
            self.reasoning = ""

    class FakePlan:
        plan_name = "test_plan"
        steps = [
            FakeStep("configure_preprocessing", {"scaler": "minmax", "onehot_max_cats": 10}),
            FakeStep("configure_cv", {"n_splits": 5, "strategy": "kfold"}),
            FakeStep("configure_scoring", {"primary_metric": "accuracy"}),
            FakeStep("configure_candidate_models", {"models": [{"name": "RandomForest"}]}),
        ]

    scope = ResolvedTargetScope(
        target_columns=["y"],
        task_family="binary_classification",
        target_mode="single_target",
        prediction_kind="label",
        primary_metric="roc_auc",
        candidate_metrics=[],
        id_columns=[],
        output_columns=["y"],
        submission_columns=["y"],
    )
    cfg = _apply_plan_to_config(FakePlan(), scope, plan_applied=True)
    assert isinstance(cfg, ModelingConfig)
    assert cfg.preprocessing["scaler"] == "minmax"
    assert cfg.preprocessing["onehot_max_cats"] == 10
    assert cfg.cv["n_splits"] == 3
    assert cfg.cv["strategy"] == "stratified_kfold"
    assert cfg.scoring["primary_metric"] == "accuracy"
    assert cfg.candidate_models is not None
    assert any(m["name"] == "RandomForest" for m in cfg.candidate_models)


def test_apply_plan_to_config_unwraps_llm_dict_payloads():
    """LLM planner sometimes wraps primitives in dict (e.g. {"strategy": "median"})."""
    from app.agents.model_building import ModelingConfig, ResolvedTargetScope, _apply_plan_to_config

    class FakeStep:
        def __init__(self, tool, parameters):
            self.tool = tool
            self.parameters = parameters
            self.reasoning = ""

    class FakePlan:
        plan_name = "wrapped_plan"
        steps = [
            FakeStep("configure_preprocessing", {
                "numeric_impute": {"strategy": "median"},
                "scaler": {"name": "standard"},
            }),
            FakeStep("configure_cv", {"strategy": {"type": "kfold"}, "n_splits": 4}),
            FakeStep("configure_scoring", {
                "primary_metric": {"name": "f1_macro"},
                "secondary_metrics": ["accuracy"],
            }),
        ]

    scope = ResolvedTargetScope(
        target_columns=["y"],
        task_family="multiclass_classification",
        target_mode="single_target",
        prediction_kind="label",
        primary_metric="f1_macro",
        candidate_metrics=[],
        id_columns=[],
        output_columns=["y"],
        submission_columns=["y"],
    )
    cfg = _apply_plan_to_config(FakePlan(), scope, plan_applied=True)
    assert isinstance(cfg, ModelingConfig)
    assert cfg.preprocessing["numeric_impute"] == "median"
    assert cfg.preprocessing["scaler"] == "standard"
    assert cfg.cv["strategy"] == "stratified_kfold"
    assert cfg.cv["n_splits"] == 3
    assert cfg.scoring["primary_metric"] == "f1_macro"
    assert cfg.scoring["secondary_metrics"] == []


def test_classify_multi_target_subtype_single_column():
    from app.agents.model_building import (
        MULTI_TARGET_SUBTYPE_SINGLE,
        _classify_multi_target_subtype,
    )
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    assert _classify_multi_target_subtype(
        df, ["y"], task_family="binary_classification", target_mode="single_target"
    ) == MULTI_TARGET_SUBTYPE_SINGLE


def test_classify_multi_target_subtype_binary_multilabel():
    from app.agents.model_building import (
        MULTI_TARGET_SUBTYPE_BINARY_MULTILABEL,
        _classify_multi_target_subtype,
    )
    df = pd.DataFrame({"a": [1, 2, 3], "t1": [0, 1, 0], "t2": [1, 0, 1]})
    assert _classify_multi_target_subtype(
        df, ["t1", "t2"],
        task_family="multilabel_classification", target_mode="multi_label",
    ) == MULTI_TARGET_SUBTYPE_BINARY_MULTILABEL


def test_classify_multi_target_subtype_multiclass_multioutput():
    from app.agents.model_building import (
        MULTI_TARGET_SUBTYPE_MULTICLASS_MULTIOUTPUT,
        _classify_multi_target_subtype,
    )
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "t1": [0, 1, 0, 1],
        "t2": [0, 1, 2, 1],  # 3 classes → triggers multiclass-multioutput
    })
    assert _classify_multi_target_subtype(
        df, ["t1", "t2"],
        task_family="multilabel_classification", target_mode="multi_label",
    ) == MULTI_TARGET_SUBTYPE_MULTICLASS_MULTIOUTPUT


def test_classify_multi_target_subtype_multioutput_regression():
    from app.agents.model_building import (
        MULTI_TARGET_SUBTYPE_MULTI_OUTPUT_REGRESSION,
        _classify_multi_target_subtype,
    )
    df = pd.DataFrame({
        "a": [1, 2, 3], "t1": [1.0, 2.0, 3.0], "t2": [4.0, 5.0, 6.0]
    })
    assert _classify_multi_target_subtype(
        df, ["t1", "t2"],
        task_family="regression", target_mode="multi_output",
    ) == MULTI_TARGET_SUBTYPE_MULTI_OUTPUT_REGRESSION


def test_apply_plan_to_config_drops_unrecoverable_dict_values():
    """When a dict value cannot be unwrapped to a primitive, keep the existing default."""
    from app.agents.model_building import ResolvedTargetScope, _apply_plan_to_config

    class FakeStep:
        def __init__(self, tool, parameters):
            self.tool = tool
            self.parameters = parameters
            self.reasoning = ""

    class FakePlan:
        plan_name = "junk_plan"
        steps = [
            FakeStep("configure_preprocessing", {
                "numeric_impute": {"unrelated": [1, 2, 3]},
            }),
        ]

    scope = ResolvedTargetScope(
        target_columns=["y"],
        task_family="binary_classification",
        target_mode="single_target",
        prediction_kind="label",
        primary_metric="roc_auc",
        candidate_metrics=[],
        id_columns=[],
        output_columns=["y"],
        submission_columns=["y"],
    )
    cfg = _apply_plan_to_config(FakePlan(), scope, plan_applied=True)
    # Default ("median") preserved when value is an unrecognizable dict.
    assert cfg.preprocessing["numeric_impute"] == "median"


def test_filter_candidates_selects_by_name():
    from app.agents.model_building import _filter_candidates
    from app.tools.model_tools import get_candidate_models

    base = get_candidate_models("binary_classification")
    choices = [{"name": "RandomForest", "param_grid": None}]
    filtered = _filter_candidates(base, choices)
    assert "RandomForest" in filtered
    assert "LogisticRegression" not in filtered


def test_filter_candidates_matches_multioutput_suffix():
    from app.agents.model_building import _filter_candidates
    from app.tools.model_tools import get_candidate_models

    base = get_candidate_models("multilabel_classification", "multi_label")
    choices = [{"name": "RandomForest", "param_grid": None}]
    filtered = _filter_candidates(base, choices)
    assert "MultiOutput_RandomForest" in filtered


def test_importance_vector_coef(binary_result):
    """Linear models expose coef_ — must return non-None importance vector."""
    from app.agents.model_building import _importance_vector_from_estimator
    from sklearn.linear_model import Ridge

    # Ridge has 1D coef_ → "coef_abs"
    ridge = Ridge()
    X = np.random.default_rng(0).random((40, 3))
    y = np.random.default_rng(0).random(40)
    ridge.fit(X, y)
    vec, method = _importance_vector_from_estimator(ridge)
    assert vec is not None
    assert method == "coef_abs"
    assert len(vec) == 3


def test_importance_vector_nested_estimators():
    """MultiOutputClassifier exposes estimators_ — multioutput_mean_importance."""
    from app.agents.model_building import _importance_vector_from_estimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    base = LogisticRegression(max_iter=200, random_state=0)
    moc = MultiOutputClassifier(base)
    X = np.random.default_rng(1).random((60, 4))
    Y = np.random.default_rng(1).integers(0, 2, (60, 3))
    moc.fit(X, Y)
    vec, method = _importance_vector_from_estimator(moc)
    assert vec is not None
    assert method == "multioutput_mean_importance"
    assert len(vec) == 4


def test_clustering_task(tmp_path):
    from app.schemas.state_schema import FeatureState, TargetSpec

    train = _clustering_train()
    train_path = tmp_path / "cluster_train.csv"
    train.to_csv(train_path, index=False)

    state = GraphState(
        target_spec=TargetSpec(
            status="resolved",
            target_columns=[],           # unsupervised: no target
            target_mode="single_target",
            task_family="clustering",
        ),
        feature_state=FeatureState(
            engineered_train_path=str(train_path),
            engineered_test_path="",
            selected_features=["feat_a", "feat_b"],
        ),
    )
    result = _run(state)
    assert "runtime_state" not in result or not result.get("runtime_state", {}).get("interrupted")
    ms = result["model_state"]
    assert ms["best_model_name"] == "KMeans"
    assert ms["validation_metrics"]["_meta"]["task_family"] == "clustering"
    # Cluster label output written to prediction file
    assert result["artifacts"]["prediction_file"]
    sub = pd.read_csv(result["artifacts"]["prediction_file"])
    assert "cluster" in sub.columns
    assert sub["cluster"].nunique() >= 2
