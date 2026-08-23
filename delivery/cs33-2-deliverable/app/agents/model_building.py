"""Model Building node.

Responsibilities:
  1. Resolve the target contract from ``state.target_spec``, with layered
     fallbacks through compatibility mirror fields and y-based inference.
  2. Read the engineered train/test CSV artifacts produced by feature engineering.
  3. Call ``BasePlanner.plan_model_building`` to produce a ``ToolCallPlan``.
     Any ``configure_*`` steps are merged into ``ModelingConfig``. If the LLM
     call fails, a task-family-aware default plan is used instead.
  4. Delegate preprocessing construction, candidate model training and
     selection, submission generation, and task-family-aware baseline metrics
     to ``app/tools/model_tools.py``.
  5. Write results back to ``ModelState`` and ``artifacts`` according to
     ``NODE_CONTRACTS["model_building"]``.

Supported task_family × target_mode combinations:
  - binary_classification × single_target
  - multiclass_classification × single_target
  - regression × single_target
  - multilabel_classification × multi_label
  - regression × multi_output  (MultiOutputRegressor)
  - clustering × single_target  (X-only, unsupervised KMeans/DBSCAN)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.state_schema import GraphState
from app.tools.metric_tools import compute_metric_bundle
from app.tools.model_tools import (
    BEST_MODEL_PATH,
    MODEL_RESULTS_PATH,
    PREDICTION_PATH,
    build_cv,
    build_preprocessor,
    build_submission,
    canonicalize_metric_name,
    default_primary_metric,
    detect_feature_types,
    fit_and_select_clustering_model,
    fit_and_select_model,
    get_candidate_models,
    infer_task_family,
    metric_direction,
    normalize_task_family,
    optimize_candidate_search_space,
    predict_for_submission,
    save_model_artifacts,
    sanitize_candidate_metrics,
    summarize_cv_results,
)
from app.tools.phase_trace import (
    phase_record_path,
    register_phase_artifact,
    resolve_phase_plan,
    write_phase_record,
)
from app.tools.run_manifest import write_run_manifest

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Resolved target scope (lightweight modeling-internal contract)
# ──────────────────────────────────────────────


# Modeling-internal subtypes for multi-column targets. These extend the
# coarse `target_mode` channel (kept stable for the rest of the project)
# with the actual per-column character of y, so the modeling phase can
# route to the correct scorer/metric path without changing schema.
#
#   single_target          — exactly one target column (no per-column logic).
#   binary_multilabel      — every target column has values ⊆ {0, 1}.
#                            Sklearn label_type == "multilabel-indicator".
#                            f1_micro / hamming_loss / roc_auc_macro_ovr OK.
#   multiclass_multioutput — at least one target column has 3+ classes.
#                            Sklearn label_type == "multiclass-multioutput".
#                            Many multilabel scorers reject this; we route
#                            to subset-accuracy + per-target macro-F1 mean.
#   multi_output_regression — target_mode == "multi_output", continuous y.
MULTI_TARGET_SUBTYPE_SINGLE = "single_target"
MULTI_TARGET_SUBTYPE_BINARY_MULTILABEL = "binary_multilabel"
MULTI_TARGET_SUBTYPE_MULTICLASS_MULTIOUTPUT = "multiclass_multioutput"
MULTI_TARGET_SUBTYPE_MULTI_OUTPUT_REGRESSION = "multi_output_regression"


def _classify_multi_target_subtype(
    train: pd.DataFrame,
    target_columns: list[str],
    *,
    task_family: str,
    target_mode: str,
) -> str:
    """Return the modeling-internal subtype for the resolved target scope.

    Inspects the actual training values rather than relying on upstream
    target_mode alone, so a "multi_label" run whose targets are not all
    binary can still be modelled correctly.
    """
    if not target_columns or len(target_columns) == 1:
        return MULTI_TARGET_SUBTYPE_SINGLE
    if target_mode == "multi_output":
        return MULTI_TARGET_SUBTYPE_MULTI_OUTPUT_REGRESSION

    binary_set = {0, 1, True, False, "0", "1"}
    all_binary = True
    for col in target_columns:
        if col not in train.columns:
            all_binary = False
            break
        series = train[col].dropna()
        if series.empty:
            all_binary = False
            break
        unique_vals = set(series.unique().tolist())
        if not unique_vals.issubset(binary_set):
            all_binary = False
            break
        if series.nunique() > 2:
            all_binary = False
            break
    if all_binary:
        return MULTI_TARGET_SUBTYPE_BINARY_MULTILABEL
    return MULTI_TARGET_SUBTYPE_MULTICLASS_MULTIOUTPUT


@dataclass
class ResolvedTargetScope:
    """Canonical resolved target contract used inside modeling, with fallbacks."""

    target_columns: list[str]
    task_family: str
    target_mode: str
    prediction_kind: str
    primary_metric: str
    candidate_metrics: list[str]
    id_columns: list[str]
    output_columns: list[str]
    submission_columns: list[str]
    multi_target_subtype: str = MULTI_TARGET_SUBTYPE_SINGLE
    warnings: list[str] = field(default_factory=list)

    def to_planner_summary(self) -> dict[str, Any]:
        return {
            "target_columns": list(self.target_columns),
            "task_family": self.task_family,
            "target_mode": self.target_mode,
            "prediction_kind": self.prediction_kind,
            "primary_metric": self.primary_metric,
            "candidate_metrics": list(self.candidate_metrics),
            "id_columns": list(self.id_columns),
            "output_columns": list(self.output_columns),
            "submission_columns": list(self.submission_columns),
            "multi_target_subtype": self.multi_target_subtype,
        }


@dataclass
class ModelingConfig:
    """Merged configuration derived from LLM plan + defaults."""

    preprocessing: dict[str, Any]
    cv: dict[str, Any]
    scoring: dict[str, Any]
    candidate_models: list[dict[str, Any]] | None
    plan_name: str
    plan_applied: bool


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────


async def run_model_building(state: GraphState) -> dict:
    """Execute the Model Building phase and return partial state update."""

    train_path = state.feature_state.engineered_train_path
    test_path = state.feature_state.engineered_test_path

    # ── 1. Path validation ───────────────────────────────────────
    if not train_path or not Path(train_path).exists():
        return _error_envelope(
            phase="model_building",
            error_type="artifact_missing",
            message=f"engineered train CSV not found: '{train_path}'",
            state=state,
        )

    has_external_test = bool(test_path and Path(test_path).exists())
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if has_external_test else pd.DataFrame()

    # ── 2. Resolve target scope (target_spec + fallbacks) ────────
    try:
        scope = _resolve_target_scope(state, train=train, test=test)
    except _TargetResolutionError as err:
        return _error_envelope(
            phase="model_building",
            error_type="schema_error",
            message=str(err),
            state=state,
        )

    data_quality_flags: list[str] = list(state.data_state.data_quality_flags)
    data_quality_flags.extend(scope.warnings)

    # ── 3. Split X / y ──────────────────────────────────────────
    X_train = train.drop(columns=scope.target_columns, errors="ignore")
    y_train = _extract_y(train, scope)

    X_test = test.drop(
        columns=[c for c in scope.target_columns if c in test.columns],
        errors="ignore",
    ) if has_external_test else None
    X_train, X_test, dropped_feature_warnings = _drop_all_missing_feature_columns(
        X_train,
        X_test,
    )
    data_quality_flags.extend(dropped_feature_warnings)

    # ── 4. Load current phase planner plan (no internal replanning) ─
    plan, plan_meta = resolve_phase_plan(
        state,
        phase="model_building",
        fallback_plan_name=f"model_building_default_{scope.task_family}",
    )
    plan_applied = plan_meta["plan_source"] == "planner_execution_plan"
    if not plan_applied:
        data_quality_flags.append(
            "model_building used deterministic fallback plan: "
            + str(plan_meta.get("fallback_reason") or "missing planner plan")
        )

    config = _apply_plan_to_config(plan, scope, plan_applied=plan_applied)

    if scope.task_family == "clustering":
        # Clustering metrics (silhouette_score etc.) are unsupported by task_contract
        # canonicalization; preserve the scope-resolved primary metric directly.
        config.scoring["primary_metric"] = scope.primary_metric
        config.scoring["secondary_metrics"] = []
    else:
        config.scoring["primary_metric"] = canonicalize_metric_name(
            scope.task_family,
            str(config.scoring.get("primary_metric") or scope.primary_metric),
        )
        config.scoring["secondary_metrics"] = []

    # ── 5. Build preprocessor ──────────────────────────────────
    selected_features = _resolved_selected_features(
        state,
        X_train,
        scope,
    )
    if X_train.shape[1] <= 0:
        return _error_envelope(
            phase="model_building",
            error_type="schema_error",
            message="no feature columns remain after target exclusion and preprocessing guards",
            state=state,
        )
    num_features, cat_features = detect_feature_types(
        X_train,
        selected_features=selected_features,
        exclude=list(scope.target_columns) + list(scope.id_columns),
    )
    preprocessor = build_preprocessor(num_features, cat_features, **config.preprocessing)

    # ── 6. Build candidate models ──────────────────────────────
    base_candidates = get_candidate_models(
        task_family=scope.task_family,
        target_mode=scope.target_mode,
        prediction_kind=scope.prediction_kind,
    )
    candidates = _filter_candidates(base_candidates, config.candidate_models)
    if not candidates:
        candidates = base_candidates   # defensive
    candidates, search_budget_notes = optimize_candidate_search_space(
        candidates,
        n_samples=int(len(X_train)),
        n_features=int(len(X_train.columns)),
        task_family=scope.task_family,
    )
    data_quality_flags.extend(search_budget_notes)

    # ── 7. CV splitter (not used for clustering) ───────────────
    cv = build_cv(scope.task_family, **config.cv)

    # ── 8. Fit + select ────────────────────────────────────────
    if scope.task_family == "clustering":
        best_pipeline, best_name, all_results = fit_and_select_clustering_model(
            X_train,
            preprocessor,
            candidates,
        )
    else:
        best_pipeline, best_name, all_results = fit_and_select_model(
            X_train,
            y_train,
            preprocessor,
            candidates,
            task_family=scope.task_family,
            target_mode=scope.target_mode,
            cv=cv,
            primary_metric=config.scoring["primary_metric"],
            multi_target_subtype=scope.multi_target_subtype,
        )
    if best_pipeline is None:
        failure_details: list[str] = []
        for cand_name, info in (all_results or {}).items():
            err = info.get("error") if isinstance(info, dict) else None
            if err:
                failure_details.append(f"{cand_name}: {str(err)[:240]}")
            else:
                failure_details.append(f"{cand_name}: <no result>")
        cause_summary = "; ".join(failure_details) or "no candidate produced a pipeline"
        diagnostic_flags = [
            f"model_building candidate failed -> {detail}"
            for detail in failure_details
        ] or [f"model_building all candidates failed: {list(all_results)}"]
        return _error_envelope(
            phase="model_building",
            error_type="runtime_error",
            message=f"all candidate models failed [{cause_summary}]",
            state=state,
            extra_quality_flags=data_quality_flags + diagnostic_flags,
        )

    # ── 9. Predict & build submission ─────────────────────────
    submission_df = None
    prediction_path = ""
    prediction_artifact_path = str(
        state.artifacts.get("prediction_file") or PREDICTION_PATH
    )
    predict_on = X_test if X_test is not None else (
        X_train if scope.task_family == "clustering" else None
    )
    if predict_on is not None:
        try:
            preds_df = predict_for_submission(
                best_pipeline,
                predict_on,
                task_family=scope.task_family,
                target_mode=scope.target_mode,
                prediction_kind=scope.prediction_kind,
                output_columns=scope.output_columns,
                multi_target_subtype=scope.multi_target_subtype,
            )
            ref_df = test if X_test is not None else train
            submission_df = build_submission(
                ref_df,
                preds_df,
                id_columns=scope.id_columns,
                output_columns=scope.output_columns,
                submission_columns=scope.submission_columns,
            )
            prediction_path = prediction_artifact_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("prediction/submission assembly failed: %s", exc)
            data_quality_flags.append(f"submission assembly failed: {exc}")

    # ── 10. Baseline metrics + explanations ────────────────────
    baseline_metrics = {}
    if scope.task_family != "clustering" and (
        scope.multi_target_subtype != MULTI_TARGET_SUBTYPE_MULTICLASS_MULTIOUTPUT
    ):
        # Skip baseline for true multiclass-multioutput: out-of-fold dummy
        # cross_val_predict on (n_samples, k) with multiple class cardinalities
        # is fragile across sklearn versions. The selected model's CV scores
        # in validation_metrics already provide the comparison signal we need.
        baseline_metrics = _compute_baseline_reference_metrics(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            cv=cv,
            scope=scope,
        )
    feature_importance_summary, model_explanations = _extract_model_explanations(
        best_pipeline,
        selected_features,
        task_family=scope.task_family,
        target_mode=scope.target_mode,
    )

    # ── 11. Summarize + persist ────────────────────────────────
    summary = summarize_cv_results(
        list(candidates.keys()),
        best_name,
        all_results,
        task_family=scope.task_family,
        target_mode=scope.target_mode,
        primary_metric=config.scoring["primary_metric"],
    )
    model_artifact_paths = save_model_artifacts(
        best_pipeline,
        summary,
        submission_df,
        model_results_path=str(state.artifacts.get("model_results") or MODEL_RESULTS_PATH),
        best_model_path=str(state.artifacts.get("best_model") or BEST_MODEL_PATH),
        prediction_path=prediction_artifact_path,
    )

    # ── 12. Build state update ─────────────────────────────────
    validation_metrics = {
        **summary["validation_metrics"],
        "_meta": {
            "task_family": scope.task_family,
            "target_mode": scope.target_mode,
            "prediction_kind": scope.prediction_kind,
            "primary_metric": config.scoring["primary_metric"],
            "metric_direction": metric_direction(config.scoring["primary_metric"]),
            "plan_name": config.plan_name,
            "plan_applied": config.plan_applied,
            "n_samples_train": int(len(train)),
            "n_features_train": int(len(X_train.columns)),
        },
    }

    plan_record = {
        "phase": "model_building",
        "plan_name": plan.plan_name,
        "plan_source": plan_meta.get("plan_source"),
        "fallback_reason": plan_meta.get("fallback_reason"),
        "criteria": plan_meta.get("criteria") or [],
        "allowed_tools": plan_meta.get("allowed_tools") or [],
        "target_scope": scope.to_planner_summary(),
        "steps": [
            {
                "tool": step.tool,
                "parameters": dict(step.parameters),
                "reasoning": step.reasoning,
            }
            for step in plan.steps
        ],
    }
    plan_record_file = phase_record_path(
        workspace_root=state.runtime_state.workspace_root,
        artifact_root=state.runtime_state.artifact_root,
        category="phase_plans",
        phase="model_building",
    )
    write_phase_record(plan_record_file, plan_record)

    updated_feature_state = {
        **state.feature_state.model_dump(),
        "feature_importance_summary": feature_importance_summary,
    }
    updated_model_state = {
        **state.model_state.model_dump(),
        "candidate_models": list(candidates.keys()),
        "best_model_name": best_name,
        "best_params": all_results.get(best_name, {}).get("best_params", {}),
        "validation_metrics": validation_metrics,
        "cv_results": all_results,
        "prediction_path": prediction_path,
        "baseline_metrics": baseline_metrics,
        "model_explanations": model_explanations,
    }

    manifest_path = str(state.artifacts.get("run_manifest") or "")
    if manifest_path:
        write_run_manifest(
            workspace_root=state.runtime_state.workspace_root,
            manifest_path=manifest_path,
            run_id=state.runtime_state.run_id or state.task_config.task_id,
            thread_id=state.runtime_state.thread_id,
            current_phase="model_building",
            phase_history=list(state.phase_history or []) + ["model_building"],
            task_config=state.task_config.model_dump(mode="python"),
            target_spec=state.target_spec.model_dump(mode="python"),
            artifacts={
                **state.artifacts,
                "model_results": model_artifact_paths["model_results"],
                "best_model": model_artifact_paths["best_model"],
                "prediction_file": model_artifact_paths["prediction_file"],
                "phase_plan_records": {
                    **dict(state.artifacts.get("phase_plan_records") or {}),
                    "model_building": plan_record_file,
                },
            },
            runtime_state=state.runtime_state.model_dump(mode="python"),
            data_state={
                **state.data_state.model_dump(mode="python"),
                "data_quality_flags": data_quality_flags,
            },
            feature_state=updated_feature_state,
            model_state=updated_model_state,
            review_state=state.review_state.model_dump(mode="python"),
            report_state=state.report_state.model_dump(mode="python"),
        )

    return {
        "feature_state": updated_feature_state,
        "model_state": updated_model_state,
        "data_state": {
            **state.data_state.model_dump(),
            "data_quality_flags": data_quality_flags,
        },
        "artifacts": register_phase_artifact(
            {
                **state.artifacts,
            "model_results": model_artifact_paths["model_results"],
            "best_model": model_artifact_paths["best_model"],
            "prediction_file": model_artifact_paths["prediction_file"],
            "run_manifest": manifest_path or state.artifacts.get("run_manifest", ""),
            },
            bucket="phase_plan_records",
            phase="model_building",
            path=plan_record_file,
        ),
        "current_phase": "model_building",
        "phase_history": ["model_building"],
    }


# ──────────────────────────────────────────────
# Target scope resolution (layered fallback)
# ──────────────────────────────────────────────


class _TargetResolutionError(Exception):
    """Target columns cannot be resolved even after all fallbacks."""


def _resolve_target_scope(
    state: GraphState,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> ResolvedTargetScope:
    warnings: list[str] = []

    # 1. Target columns: target_spec → compatibility mirrors
    target_columns = list(state.target_spec.target_columns or [])
    if not target_columns:
        mirrors = (
            list(state.problem_understanding.target_columns or [])
            or list(state.data_state.target_columns or [])
            or list(state.task_config.target_columns or [])
        )
        if not mirrors:
            single = (
                state.problem_understanding.target_column
                or state.data_state.target_column
                or state.task_config.target_column
                or ""
            ).strip()
            if single:
                mirrors = [single]
        if mirrors:
            warnings.append(
                "target_spec.target_columns empty; fell back to legacy mirror fields"
            )
            target_columns = mirrors
    _family_hint = normalize_task_family(state.target_spec.task_family or "")
    if not target_columns and _family_hint != "clustering":
        raise _TargetResolutionError(
            "target_spec.target_columns is empty and no legacy mirror provides one"
        )

    missing = [c for c in target_columns if c not in train.columns]
    if missing:
        raise _TargetResolutionError(
            f"target columns not present in training data: {missing}; "
            f"available columns: {list(train.columns)}"
        )

    # For clustering, task_family is resolved early so downstream branches can detect it.
    if _family_hint == "clustering":
        return ResolvedTargetScope(
            target_columns=[],
            task_family="clustering",
            target_mode="single_target",
            prediction_kind="label",
            primary_metric="silhouette_score",
            candidate_metrics=[],
            id_columns=[],
            output_columns=["cluster"],
            submission_columns=["cluster"],
            warnings=["clustering task: no target columns required"],
        )

    # 2. target_mode (normalize legacy 'single' / 'multi' to plan vocabulary)
    target_mode_raw = (state.target_spec.target_mode or "").strip().lower() or "unknown"
    target_mode = {
        "single": "single_target",
        "multi": "multi_label",   # conservative default; refined by y below
    }.get(target_mode_raw, target_mode_raw)
    if target_mode in {"unknown", ""}:
        if len(target_columns) == 1:
            target_mode = "single_target"
        else:
            if all(train[c].dropna().nunique() <= 2 for c in target_columns):
                target_mode = "multi_label"
            else:
                target_mode = "multi_output"
        warnings.append(f"target_mode inferred as '{target_mode}'")

    # 3. task_family (canonical)
    raw_family = state.target_spec.task_family or ""
    task_family = normalize_task_family(raw_family)
    if task_family in {"unknown", "classification_unspecified"}:
        inferred = infer_task_family(
            train[target_columns] if len(target_columns) > 1 else train[target_columns[0]],
            target_mode=target_mode,
        )
        warnings.append(
            f"task_family inferred as '{inferred}' "
            f"(upstream target_spec.task_family='{raw_family or 'empty'}')"
        )
        task_family = inferred

    # Harmonize family with target_mode
    if target_mode == "multi_label" and task_family != "multilabel_classification":
        warnings.append(
            f"target_mode=multi_label but task_family={task_family}; "
            f"overriding to multilabel_classification"
        )
        task_family = "multilabel_classification"
    if target_mode == "multi_output" and task_family != "regression":
        warnings.append(
            f"target_mode=multi_output is only supported for regression; "
            f"task_family={task_family} — overriding target_mode to single_target"
        )
        target_columns = target_columns[:1]
        target_mode = "single_target"

    # 4. prediction_kind
    prediction_kind = (
        state.target_spec.prediction_kind or _default_prediction_kind(task_family)
    ).strip()

    # 5. primary_metric + candidate_metrics
    primary_metric = (
        state.target_spec.evaluation.primary_metric
        or default_primary_metric(task_family)
    )
    primary_metric = canonicalize_metric_name(task_family, primary_metric)
    candidate_metrics = sanitize_candidate_metrics(
        task_family,
        list(state.target_spec.evaluation.candidate_metrics or []),
    )

    # 6. Submission spec: id_columns / output_columns / submission_columns
    id_columns = list(state.target_spec.submission.id_columns or [])
    if not id_columns and not test.empty:
        guessed = _guess_id_columns(test, target_columns)
        if guessed:
            warnings.append(f"id_columns inferred as {guessed}")
            id_columns = guessed

    output_columns = list(state.target_spec.submission.output_columns or [])
    if not output_columns:
        output_columns = list(target_columns)
        warnings.append("output_columns defaulted to target_columns")

    submission_columns = list(state.target_spec.submission.submission_columns or [])
    if not submission_columns:
        submission_columns = list(id_columns) + list(output_columns)

    multi_target_subtype = _classify_multi_target_subtype(
        train,
        target_columns,
        task_family=task_family,
        target_mode=target_mode,
    )
    if (
        target_mode == "multi_label"
        and multi_target_subtype == MULTI_TARGET_SUBTYPE_MULTICLASS_MULTIOUTPUT
    ):
        warnings.append(
            "multi-target detected as multiclass-multioutput (≥3 classes in some "
            "column); modeling will use subset-accuracy + per-target macro-F1 path"
        )

    return ResolvedTargetScope(
        target_columns=target_columns,
        task_family=task_family,
        target_mode=target_mode,
        prediction_kind=prediction_kind,
        primary_metric=primary_metric,
        candidate_metrics=candidate_metrics,
        id_columns=id_columns,
        output_columns=output_columns,
        submission_columns=submission_columns,
        multi_target_subtype=multi_target_subtype,
        warnings=warnings,
    )


def _default_prediction_kind(task_family: str) -> str:
    if task_family == "regression":
        return "continuous"
    return "label"


def _guess_id_columns(test: pd.DataFrame, target_columns: list[str]) -> list[str]:
    """Heuristic id-column guess for use when target_spec doesn't specify.

    Priority:
      1. Columns matching common Kaggle id names (Id, id, PassengerId, index).
      2. First column that is not a target column and not present in training
         feature space — leave empty if no good match.
    """
    preferred = {"id", "Id", "ID", "PassengerId", "index", "Index"}
    for col in test.columns:
        if col in preferred and col not in target_columns:
            return [col]
    # Fall back: single non-target unique-valued column
    for col in test.columns:
        if col in target_columns:
            continue
        if test[col].is_unique:
            return [col]
    return []


def _extract_y(train: pd.DataFrame, scope: ResolvedTargetScope):
    if not scope.target_columns:
        return None  # clustering: X-only
    if scope.target_mode in {"multi_label", "multi_output"} and len(scope.target_columns) > 1:
        return train[scope.target_columns]
    return train[scope.target_columns[0]]


def _drop_all_missing_feature_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Drop columns that are entirely missing in training data."""

    warnings: list[str] = []
    all_missing = [column for column in X_train.columns if X_train[column].isna().all()]
    if not all_missing:
        return X_train, X_test, warnings
    warnings.append(
        "dropped all-missing feature columns before preprocessing: "
        + ", ".join(all_missing)
    )
    X_train = X_train.drop(columns=all_missing, errors="ignore")
    if X_test is not None:
        X_test = X_test.drop(columns=all_missing, errors="ignore")
    return X_train, X_test, warnings


def _resolved_selected_features(
    state: GraphState,
    X_train: pd.DataFrame,
    scope: ResolvedTargetScope,
) -> list[str]:
    selected = [
        column
        for column in list(state.feature_state.selected_features or [])
        if column in X_train.columns and column not in set(scope.id_columns)
    ]
    if selected:
        return selected
    return [column for column in X_train.columns if column not in set(scope.id_columns)]


# ──────────────────────────────────────────────
# Plan → ModelingConfig merge (with task-family defaults)
# ──────────────────────────────────────────────

_PREPROC_KEYS = {"numeric_impute", "scaler", "cat_encode", "cat_impute", "onehot_max_cats"}
_SCORING_KEYS = {"primary_metric"}

# LLM planner sometimes wraps a single-choice value in a dict, e.g.
# numeric_impute={"strategy": "median"} instead of "median". When we see a dict
# at a position that should be a primitive, fall back to one of these inner
# keys before discarding the value entirely.
_PRIMITIVE_DICT_UNWRAP_KEYS = ("strategy", "name", "type", "value", "method")


def _coerce_primitive_param(value: Any) -> Any:
    """Unwrap LLM-style {"strategy": <primitive>} payloads to the inner primitive.

    Returns the original value when it already looks primitive (str/int/float/bool/None)
    or when no wrapper key matched. Caller is responsible for substituting a
    default when the result is still a non-primitive container.
    """
    if isinstance(value, dict):
        for inner_key in _PRIMITIVE_DICT_UNWRAP_KEYS:
            inner = value.get(inner_key)
            if isinstance(inner, (str, int, float, bool)):
                return inner
    return value


def _apply_plan_to_config(
    plan: ToolCallPlan,
    scope: ResolvedTargetScope,
    *,
    plan_applied: bool,
) -> ModelingConfig:
    preprocessing: dict[str, Any] = _default_preprocessing(scope)
    cv: dict[str, Any] = _default_cv(scope)
    scoring: dict[str, Any] = {
        "primary_metric": scope.primary_metric,
        "secondary_metrics": [],
    }
    candidate_models: list[dict[str, Any]] | None = _default_candidate_model_choices(
        scope
    )

    for step in plan.steps or []:
        tool = (step.tool or "").strip()
        params = step.parameters or {}
        if tool == "configure_preprocessing":
            for key, value in params.items():
                if key not in _PREPROC_KEYS:
                    continue
                coerced = _coerce_primitive_param(value)
                # secondary_metrics is the only PREPROC-adjacent slot that may
                # legitimately hold a list; everything else must be primitive.
                if isinstance(coerced, (str, int, float, bool)) or coerced is None:
                    preprocessing[key] = coerced
        elif tool == "configure_scoring":
            for key, value in params.items():
                if key not in _SCORING_KEYS:
                    continue
                coerced = _coerce_primitive_param(value)
                if isinstance(coerced, (str, int, float, bool)) or coerced is None:
                    scoring[key] = coerced
        elif tool == "configure_candidate_models":
            raw = params.get("models")
            if isinstance(raw, list):
                cleaned = []
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    name = str(entry.get("name") or "").strip()
                    if not name:
                        continue
                    grid = entry.get("param_grid")
                    cleaned.append({
                        "name": name,
                        "param_grid": grid if isinstance(grid, dict) else None,
                    })
                if cleaned:
                    candidate_models = cleaned[:5]

    # Final sanity: primary_metric present
    scoring.setdefault("primary_metric", scope.primary_metric)

    return ModelingConfig(
        preprocessing=preprocessing,
        cv=cv,
        scoring=scoring,
        candidate_models=candidate_models,
        plan_name=plan.plan_name or f"model_building_default_{scope.task_family}",
        plan_applied=plan_applied,
    )


def _default_preprocessing(scope: ResolvedTargetScope) -> dict[str, Any]:
    return {
        "numeric_impute": "median",
        "scaler": "standard",
        "cat_encode": "onehot",
        "cat_impute": "most_frequent",
        "onehot_max_cats": 25,
    }


def _default_cv(scope: ResolvedTargetScope) -> dict[str, Any]:
    return {
        "strategy": (
            "stratified_kfold"
            if scope.task_family
            in {"binary_classification", "multiclass_classification"}
            else "kfold"
        ),
        "n_splits": 3,
        "random_state": 42,
    }


def _default_candidate_model_choices(
    scope: ResolvedTargetScope,
) -> list[dict[str, Any]]:
    if scope.task_family == "clustering":
        names = ["KMeans", "DBSCAN"]
    elif scope.multi_target_subtype == MULTI_TARGET_SUBTYPE_MULTI_OUTPUT_REGRESSION:
        names = ["Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "LightGBM"]
    elif scope.task_family == "multilabel_classification":
        names = ["LogisticRegression", "RandomForest", "ExtraTrees", "LightGBM", "XGBoost"]
    elif "classif" in scope.task_family:
        names = [
            "LogisticRegression",
            "RandomForest",
            "ExtraTrees",
            "HistGradientBoosting",
            "LightGBM",
        ]
    else:
        names = ["Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "LightGBM"]
    return [{"name": name, "param_grid": None} for name in names[:5]]


# ──────────────────────────────────────────────
# Candidate model filtering (LLM-specified subset)
# ──────────────────────────────────────────────


def _filter_candidates(
    base: dict[str, dict],
    llm_choices: list[dict[str, Any]] | None,
) -> dict[str, dict]:
    """If LLM specified models, keep only those (and override their param_grid).

    The LLM sees base names (e.g. "RandomForest"), not the auto-wrapped
    "MultiOutput_RandomForest" — so match by suffix when wrapping is active.
    """
    if not llm_choices:
        return base

    filtered: dict[str, dict] = {}
    base_names = list(base.keys())
    for entry in llm_choices:
        requested = str(entry.get("name") or "").strip()
        if not requested:
            continue
        # Direct match
        if requested in base:
            filtered[requested] = _apply_grid_override(base[requested], entry.get("param_grid"))
            continue
        # Match suffix for wrapped multi-output (MultiOutput_<name>)
        for name in base_names:
            if name.endswith(f"_{requested}"):
                filtered[name] = _apply_grid_override(base[name], entry.get("param_grid"))
                break

    return filtered or base


def _apply_grid_override(cfg: dict, raw_grid) -> dict:
    if not isinstance(raw_grid, dict) or not raw_grid:
        return cfg
    # Build new grid using base estimator keys rewritten to pipeline-prefixed.
    # Heuristic: if existing cfg grid uses classifier__estimator__* prefix
    # (multi-output wrap), keep that prefix; otherwise use classifier__*.
    existing_grid = cfg.get("param_grid", {})
    uses_estimator_prefix = any(
        k.startswith("classifier__estimator__") for k in existing_grid
    )

    new_grid: dict[str, Any] = {}
    for key, value in raw_grid.items():
        clean_key = key
        # Strip any prefixes the LLM might have added
        for prefix in ("classifier__estimator__", "classifier__", "estimator__"):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        if not isinstance(value, list):
            value = [value]
        pipeline_key = (
            f"classifier__estimator__{clean_key}"
            if uses_estimator_prefix
            else f"classifier__{clean_key}"
        )
        new_grid[pipeline_key] = value
    return {"estimator": cfg["estimator"], "param_grid": new_grid}


# ──────────────────────────────────────────────
# predict_proba safe wrapper (task-family aware)
# ──────────────────────────────────────────────


def _safe_predict_proba(pipeline, X, task_family: str, target_mode: str):
    family = normalize_task_family(task_family)
    if family == "regression":
        return None
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        raw = pipeline.predict_proba(X)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(raw, list):  # MultiOutputClassifier
        return raw
    raw = np.asarray(raw)
    if family == "binary_classification" and raw.ndim == 2 and raw.shape[1] == 2:
        return raw[:, 1]
    return raw


def _compute_baseline_reference_metrics(
    *,
    X_train: pd.DataFrame,
    y_train,
    preprocessor,
    cv,
    scope: ResolvedTargetScope,
) -> dict[str, Any]:
    """Compute an out-of-fold dummy baseline for trustworthy comparison."""

    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.model_selection import cross_val_predict
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline

    if scope.task_family == "regression" and scope.target_mode == "multi_output":
        from sklearn.multioutput import MultiOutputRegressor

        estimator = MultiOutputRegressor(DummyRegressor(strategy="mean"))
    elif scope.task_family == "regression":
        estimator = DummyRegressor(strategy="mean")
    elif scope.target_mode == "multi_label" or scope.task_family == "multilabel_classification":
        estimator = MultiOutputClassifier(DummyClassifier(strategy="most_frequent"))
    else:
        estimator = DummyClassifier(strategy="most_frequent")

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    try:
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("baseline cross-val prediction failed: %s", exc)
        return {}

    y_prob = None
    if scope.task_family != "regression":
        try:
            y_prob = cross_val_predict(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                n_jobs=1,
                method="predict_proba",
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("baseline predict_proba unavailable: %s", exc)

    metrics = compute_metric_bundle(
        y_train,
        y_pred,
        y_prob,
        task_family=scope.task_family,
        target_mode=scope.target_mode,
        multi_target_subtype=scope.multi_target_subtype,
    )
    metrics["_meta"] = {
        "source": "out_of_fold_dummy_baseline",
        "strategy": "most_frequent"
        if scope.task_family != "regression"
        else "mean",
    }
    return metrics


def _extract_model_explanations(
    pipeline,
    selected_features: list[str],
    *,
    task_family: str,
    target_mode: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract lightweight feature-importance style explanations when available."""

    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        estimator = pipeline.named_steps.get("classifier")
    except Exception:  # noqa: BLE001
        return {}, {}

    if preprocessor is None or estimator is None:
        return {}, {}

    feature_names = _resolved_feature_names(preprocessor, selected_features)
    if not feature_names:
        return {}, {}

    importance_vector, method = _importance_vector_from_estimator(estimator)
    if importance_vector is None:
        return {}, {}

    # After OHE the preprocessor expands features; if get_feature_names_out()
    # fell back to selected_features (pre-OHE count), fix by using numeric names.
    if len(importance_vector) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(importance_vector))]

    ranked = sorted(
        zip(feature_names, importance_vector, strict=False),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    top_ranked = ranked[:10]
    feature_importance_summary = {
        name: round(float(score), 6) for name, score in top_ranked
    }
    model_explanations = {
        "importance_method": method,
        "task_family": task_family,
        "target_mode": target_mode,
        "top_features": [
            {"feature": name, "score": round(float(score), 6)}
            for name, score in top_ranked[:5]
        ],
    }
    return feature_importance_summary, model_explanations


def _resolved_feature_names(preprocessor, selected_features: list[str]) -> list[str]:
    try:
        raw = preprocessor.get_feature_names_out()
        names = [str(name) for name in raw]
        return names or list(selected_features)
    except Exception:  # noqa: BLE001
        return list(selected_features)


def _importance_vector_from_estimator(estimator) -> tuple[np.ndarray | None, str]:
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float), "feature_importances"

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            return np.abs(coef), "coef_abs"
        return np.mean(np.abs(coef), axis=0), "coef_abs_mean"

    nested_estimators = getattr(estimator, "estimators_", None)
    if nested_estimators:
        vectors: list[np.ndarray] = []
        for inner in nested_estimators:
            vector, _ = _importance_vector_from_estimator(inner)
            if vector is not None:
                vectors.append(np.asarray(vector, dtype=float))
        if vectors:
            stacked = np.vstack(vectors)
            return np.mean(np.abs(stacked), axis=0), "multioutput_mean_importance"

    return None, ""


# ──────────────────────────────────────────────
# Error envelope
# ──────────────────────────────────────────────


def _error_envelope(
    phase: str,
    error_type: str,
    message: str,
    state: "GraphState | None" = None,
    extra_quality_flags: list[str] | None = None,
) -> dict:
    error_str = f"[{phase}] {error_type}: {message}"
    logger.error(error_str)
    # Preserve existing runtime_state fields so workspace_root/artifact_root are not
    # wiped out (which would make the next retry fail at path validation in 80ms).
    # Do NOT include review_state — overwriting it resets retry_count_by_phase to {},
    # preventing the reviewer from escalating retry→fatal after 3 attempts.
    if state is None:
        return {"runtime_state": {"interrupted": True}}
    rt = state.runtime_state.model_dump(mode="python")
    rt["interrupted"] = True
    update: dict[str, Any] = {"runtime_state": rt}
    if extra_quality_flags:
        ds = state.data_state.model_dump(mode="python")
        existing_flags = list(ds.get("data_quality_flags") or [])
        for flag in extra_quality_flags:
            if flag and flag not in existing_flags:
                existing_flags.append(flag)
        ds["data_quality_flags"] = existing_flags
        update["data_state"] = ds
    return update


__all__ = ["run_model_building", "ResolvedTargetScope", "ModelingConfig"]
