"""Phase-aware planner agent for building tool-calling plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.prompts.planner_prompt import (
    PHASE_TOOLLIST_SYSTEM_PROMPT,
    build_phase_tolist_prompt,
)
from app.schemas.state_schema import ExecutionPlan, GraphState
from app.schemas.tools import ToolCall, ToolCallPlan
from app.llm.helpers import complete_json
from app.tools.profiler import profile_dataset


EXECUTION_PHASES: tuple[str, ...] = (
    "preliminary_eda",
    "data_cleaning",
    "indepth_eda",
    "feature_engineering",
    "model_building",
)


DEFAULT_PHASE_OBJECTIVES: dict[str, str] = {
    "preliminary_eda": "Profile the dataset, summarize schema, and create EDA evidence for later planning.",
    "data_cleaning": "Create cleaned train/test datasets and a traceable cleaning report.",
    "indepth_eda": "Perform in-depth EDA to uncover hidden patterns and insights.",
    "feature_engineering": "Create aligned engineered feature datasets and feature summary artifacts.",
    "model_building": "Train candidate models, select the best model, and produce prediction artifacts with metric evidence.",
}

DEFAULT_SUCCESS_CRITERIA: dict[str, list[str]] = {
    "preliminary_eda": [
        "artifacts.preliminary_eda_report exists",
        "data_state.schema_summary and data_state.eda_findings are populated",
    ],
    "data_cleaning": [
        "artifacts.cleaned_train_data and artifacts.cleaned_test_data exist",
        "data_state.current_train_path and current_test_path point to cleaned files",
    ],
    "indepth_eda": [
        "artifacts.indepth_eda_report exists",
        "data_state.schema_summary and data_state.eda_findings are populated",
    ],
    "feature_engineering": [
        "feature_state.engineered_train_path and engineered_test_path exist",
        "feature_state.selected_features is populated",
    ],
    "model_building": [
        "artifacts.model_results, artifacts.best_model, and artifacts.prediction_file exist",
        "model_state.best_model_name and validation metrics are populated",
    ],
}

PHASE_TOOL_ALLOWLISTS: dict[str, dict[str, list[str]]] = {
    "preliminary_eda": {
        "default": [
            "profile_dataset",
            "missingness_visualization",
            "basic_distribution_visualization",
        ]
    },
    "data_cleaning": {
        "default": ["drop_columns", "fill_missing", "clip_outliers"],
    },
    "indepth_eda": {
        "default": [
            "target_analysis",
            "feature_feature_relationships",
            "outlier_analysis",
            "segment_analysis",
        ]
    },
    "feature_engineering": {
        "default": [
            "extract_regex",
            "create_binning",
            "create_interaction",
            "group_rare_categories",
            "encode_categorical",
        ]
    },
    "model_building": {
        "classification": [
            "detect_feature_types",
            "build_preprocessor",
            "get_candidate_models",
            "fit_and_select_model",
            "summarize_cv_results",
            "compute_metric_bundle",
            "build_submission",
            "save_model_artifacts",
        ],
        "multilabel": [
            "detect_feature_types",
            "build_preprocessor",
            "get_candidate_models",
            "fit_and_select_model",
            "summarize_cv_results",
            "compute_metric_bundle",
            "build_submission",
            "save_model_artifacts",
        ],
        "multi_output": [
            "detect_feature_types",
            "build_preprocessor",
            "fit_and_select_model",
            "summarize_cv_results",
            "compute_metric_bundle",
            "build_submission",
            "save_model_artifacts",
        ],
        "regression": [
            "detect_feature_types",
            "build_preprocessor",
            "fit_and_select_model",
            "summarize_cv_results",
            "save_model_artifacts",
        ],
        "default": [
            "detect_feature_types",
            "build_preprocessor",
            "fit_and_select_model",
            "summarize_cv_results",
            "save_model_artifacts",
        ],
    },
}


AGENT_TOOLLIST_ALLOWLISTS: dict[str, list[str]] = {
    "data_cleaning": ["drop_columns", "fill_missing", "clip_outliers"],
    "indepth_eda": [
        "target_analysis",
        "feature_feature_relationships",
        "outlier_analysis",
        "segment_analysis",
    ],
    "feature_engineering": [
        "extract_regex",
        "create_binning",
        "create_interaction",
        "group_rare_categories",
        "encode_categorical",
    ],
    "model_building": [
        "configure_preprocessing",
        "configure_candidate_models",
        "configure_scoring",
    ],
}


PHASE_TOOL_PARAMETER_ALIASES: dict[str, dict[str, dict[str, str]]] = {
    "data_cleaning": {
        "fill_missing": {
            "columns": "column",
        },
        "clip_outliers": {
            "columns": "column",
        },
    },
    "feature_engineering": {
        "extract_regex": {
            "source_column": "column",
            "new_columns": "new_col",
        },
        "group_rare_categories": {
            "replacement_value": "rare_label",
        },
    },
    "model_building": {
        "configure_scoring": {
            "primary": "primary_metric",
            "secondary": "secondary_metrics",
        },
    },
}


PHASE_TOOL_PARAMETER_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "data_cleaning": {
        "drop_columns": {
            "required": ["columns"],
            "optional": [],
            "field_types": {
                "columns": "string | list[string]",
            },
            "example": {"columns": ["Name", "Ticket"]},
        },
        "fill_missing": {
            "required": ["column", "strategy"],
            "optional": ["fill_value", "group_cols"],
            "field_types": {
                "column": "string",
                "strategy": "string (constant|mean|median|mode)",
                "fill_value": "any",
                "group_cols": "list[string]",
            },
            "example": {"column": "Age", "strategy": "median"},
        },
        "clip_outliers": {
            "required": ["column"],
            "optional": ["lower_quantile", "upper_quantile"],
            "field_types": {
                "column": "string",
                "lower_quantile": "float",
                "upper_quantile": "float",
            },
            "example": {
                "column": "Fare",
                "lower_quantile": 0.01,
                "upper_quantile": 0.99,
            },
        },
    },
    "feature_engineering": {
        "extract_regex": {
            "required": ["column", "pattern", "new_col"],
            "optional": [],
            "field_types": {
                "column": "string",
                "pattern": "string (regex with capture group)",
                "new_col": "string",
            },
            "example": {
                "column": "Name",
                "pattern": "([A-Za-z]+)\\.",
                "new_col": "Title",
            },
        },
        "create_binning": {
            "required": ["column", "n_bins", "new_col"],
            "optional": ["strategy"],
            "field_types": {
                "column": "string",
                "n_bins": "int",
                "new_col": "string",
                "strategy": "string (quantile|uniform)",
            },
            "example": {
                "column": "Age",
                "n_bins": 5,
                "new_col": "AgeBin",
                "strategy": "quantile",
            },
        },
        "create_interaction": {
            "required": ["cols", "operator", "new_col"],
            "optional": [],
            "field_types": {
                "cols": "list[string]",
                "operator": "string (+|*|-|/)",
                "new_col": "string",
            },
            "example": {
                "cols": ["SibSp", "Parch"],
                "operator": "+",
                "new_col": "FamilySize",
            },
        },
        "group_rare_categories": {
            "required": ["column"],
            "optional": ["threshold", "rare_label"],
            "field_types": {
                "column": "string",
                "threshold": "float",
                "rare_label": "string",
            },
            "example": {"column": "Cabin", "threshold": 0.03, "rare_label": "Other"},
        },
        "encode_categorical": {
            "required": ["column", "method"],
            "optional": ["mapping"],
            "field_types": {
                "column": "string",
                "method": "string (ordinal|label|onehot)",
                "mapping": "dict[string,int]",
            },
            "example": {"column": "Embarked", "method": "onehot"},
        },
    },
    "model_building": {
        "configure_preprocessing": {
            "required": [],
            "optional": [
                "numeric_impute",
                "scaler",
                "cat_encode",
                "cat_impute",
                "onehot_max_cats",
            ],
        },
        "configure_candidate_models": {
            "required": ["models"],
            "optional": [],
        },
        "configure_scoring": {
            "required": [],
            "optional": ["primary_metric"],
        },
    },
}

DEFAULT_REVIEW_POINTS: list[str] = [
    "Reviewer must validate required artifacts for the selected phase.",
    "Reviewer must validate that phase-specific state fields are populated before advancing.",
]

DEFAULT_FALLBACK_RULES: list[str] = [
    "If reviewer requests retry, repeat the pending review phase.",
    "If LLM output is unavailable or invalid, use deterministic phase order.",
]

DETERMINISTIC_ONLY_PLANNER_PHASES: frozenset[str] = frozenset(
    {"data_cleaning", "indepth_eda"}
)


class PlannerAgent:
    """Build a phase-scoped execution plan with task-aware tool allowlists."""

    supported_phases = EXECUTION_PHASES

    async def plan_data_cleaning(
        self,
        profile_json: dict[str, Any],
        target_spec_summary: dict[str, Any] | None = None,
    ) -> ToolCallPlan:
        return await self._plan_tool_calls_for_phase(
            phase="data_cleaning",
            normalized_task_type="default",
            profile_summary=profile_json,
            target_spec_summary=target_spec_summary or {},
        )

    async def plan_feature_engineering(
        self,
        profile_json: dict[str, Any],
        target_spec_summary: dict[str, Any] | None = None,
    ) -> ToolCallPlan:
        return await self._plan_tool_calls_for_phase(
            phase="feature_engineering",
            normalized_task_type="default",
            profile_summary=profile_json,
            target_spec_summary=target_spec_summary or {},
        )

    async def plan_indepth_eda(
        self,
        profile_json: dict[str, Any],
        target_spec_summary: dict[str, Any] | None = None,
    ) -> ToolCallPlan:
        return await self._plan_tool_calls_for_phase(
            phase="indepth_eda",
            normalized_task_type="default",
            profile_summary=profile_json,
            target_spec_summary=target_spec_summary or {},
        )

    async def plan_model_building(
        self,
        profile_json: dict[str, Any],
        target_spec_summary: dict[str, Any],
    ) -> ToolCallPlan:
        normalized_task_type = self._normalized_task_type_from_target_spec(
            target_spec_summary
        )
        return await self._plan_tool_calls_for_phase(
            phase="model_building",
            normalized_task_type=normalized_task_type,
            profile_summary=profile_json,
            target_spec_summary=target_spec_summary,
        )

    async def run(self, state: GraphState) -> dict:
        """Build/update execution_plan for the current phase only."""

        target_phase = self._require_current_execution_phase(state)
        normalized_task_type = self._normalized_task_type(state)
        phase_tool_allowlist = self._phase_tool_allowlist(normalized_task_type)
        agent_plan = await self._plan_current_phase_from_state(
            state=state,
            target_phase=target_phase,
            normalized_task_type=normalized_task_type,
        )
        execution_plan = self._execution_plan_from_agent_plan(
            target_phase,
            phase_tool_allowlist,
            agent_plan,
        )

        return {
            "execution_plan": execution_plan.model_dump(mode="python"),
        }

    def _require_current_execution_phase(self, state: GraphState) -> str:
        current_phase = state.current_phase
        if current_phase in self.supported_phases:
            return current_phase
        raise ValueError(
            f"planner requires current_phase to be one of {self.supported_phases}, "
            f"got {current_phase!r}"
        )

    def _normalized_task_type(self, state: GraphState) -> str:
        task_family = (state.target_spec.task_family or "").lower()
        target_mode = (state.target_spec.target_mode or "").lower()

        return self._normalize_task_type(task_family, target_mode)

    def _normalized_task_type_from_target_spec(
        self, target_spec_summary: dict[str, Any]
    ) -> str:
        task_family = str(target_spec_summary.get("task_family") or "").lower()
        target_mode = str(target_spec_summary.get("target_mode") or "").lower()
        return self._normalize_task_type(task_family, target_mode)

    @staticmethod
    def _normalize_task_type(task_family: str, target_mode: str) -> str:
        if "multilabel" in task_family or target_mode == "multi_label":
            return "multilabel"
        if target_mode == "multi_output":
            return "multi_output"
        if "cluster" in task_family:
            return "clustering"
        if "classif" in task_family:
            return "classification"
        if "regress" in task_family:
            return "regression"
        return "default"

    async def _plan_tool_calls_for_phase(
        self,
        *,
        phase: str,
        normalized_task_type: str,
        profile_summary: dict[str, Any],
        target_spec_summary: dict[str, Any],
    ) -> ToolCallPlan:
        allowed_tools = AGENT_TOOLLIST_ALLOWLISTS.get(phase, [])
        fallback_payload = {
            "plan_name": f"{phase}_{normalized_task_type}_noop_plan",
            "steps": [],
        }

        payload = await complete_json(
            agent_name="planner",
            system_prompt=PHASE_TOOLLIST_SYSTEM_PROMPT,
            user_prompt=build_phase_tolist_prompt(
                phase=phase,
                normalized_task_type=normalized_task_type,
                profile_summary=profile_summary,
                target_spec_summary=target_spec_summary,
                retry_context=self._retry_context_for_phase(
                    phase=phase,
                    profile_summary=profile_summary,
                ),
                allowed_tools=allowed_tools,
                tool_parameter_schemas=PHASE_TOOL_PARAMETER_SCHEMAS.get(phase, {}),
            ),
            fallback=fallback_payload,
        )
        return self._validated_tool_call_plan(
            payload=payload,
            phase=phase,
            normalized_task_type=normalized_task_type,
            allowed_tools=allowed_tools,
            profile_summary=profile_summary,
            target_spec_summary=target_spec_summary,
        )

    def _validated_tool_call_plan(
        self,
        *,
        payload: dict[str, Any],
        phase: str,
        normalized_task_type: str,
        allowed_tools: list[str],
        profile_summary: dict[str, Any],
        target_spec_summary: dict[str, Any],
    ) -> ToolCallPlan:
        fallback_plan_name = f"{phase}_{normalized_task_type}_noop_plan"
        plan_name = str(payload.get("plan_name") or fallback_plan_name).strip()
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            raw_steps = []

        cleaned_steps: list[ToolCall] = []
        phase_schemas = PHASE_TOOL_PARAMETER_SCHEMAS.get(phase, {})
        phase_aliases = PHASE_TOOL_PARAMETER_ALIASES.get(phase, {})

        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            tool_name = str(step.get("tool") or "").strip()
            if not tool_name or tool_name not in allowed_tools:
                continue

            params = step.get("parameters")
            if not isinstance(params, dict):
                params = {}
            if phase == "model_building":
                params = self._normalize_model_building_parameters(
                    tool_name=tool_name,
                    params=params,
                )

            schema = phase_schemas.get(tool_name, {})
            aliases = phase_aliases.get(tool_name, {})
            params = self._normalize_tool_parameters(
                params=params,
                schema=schema,
                aliases=aliases,
            )
            if params is None:
                continue
            if not self._is_phase_step_safe(
                phase=phase,
                tool_name=tool_name,
                params=params,
                profile_summary=profile_summary,
                target_spec_summary=target_spec_summary,
                normalized_task_type=normalized_task_type,
            ):
                continue

            reasoning = str(step.get("reasoning") or "").strip() or None
            cleaned_steps.append(
                ToolCall(tool=tool_name, parameters=params, reasoning=reasoning)
            )

        return ToolCallPlan(plan_name=plan_name, steps=cleaned_steps)

    @staticmethod
    def _normalize_tool_parameters(
        *,
        params: dict[str, Any],
        schema: dict[str, Any],
        aliases: dict[str, str],
    ) -> dict[str, Any] | None:
        normalized = dict(params)

        for src, dst in aliases.items():
            if src in normalized and dst not in normalized:
                normalized[dst] = normalized.pop(src)
            elif src in normalized and dst in normalized:
                normalized.pop(src)

        required = [str(k) for k in schema.get("required", [])]
        optional = [str(k) for k in schema.get("optional", [])]
        allowed = set(required + optional)

        if allowed:
            normalized = {k: v for k, v in normalized.items() if k in allowed}

        missing_required = [k for k in required if k not in normalized]
        if missing_required:
            return None

        return normalized

    def _is_phase_step_safe(
        self,
        *,
        phase: str,
        tool_name: str,
        params: dict[str, Any],
        profile_summary: dict[str, Any],
        target_spec_summary: dict[str, Any],
        normalized_task_type: str,
    ) -> bool:
        columns = profile_summary.get("columns") or {}
        if not isinstance(columns, dict):
            columns = {}

        protected = set(target_spec_summary.get("target_columns") or []) | set(
            target_spec_summary.get("id_columns") or []
        )

        def _column_type(column_name: str) -> str:
            meta = columns.get(column_name) or {}
            return str(meta.get("type") or "").lower()

        def _exists(column_name: str) -> bool:
            return column_name in columns

        def _all_exist(column_names: list[str]) -> bool:
            return all(_exists(name) for name in column_names)

        string_like = {"categorical", "string", "text", "object"}

        if phase == "data_cleaning":
            if tool_name == "drop_columns":
                cols = params.get("columns")
                values = cols if isinstance(cols, list) else [cols] if isinstance(cols, str) else []
                return not any(col in protected for col in values)
            if tool_name == "fill_missing":
                column = params.get("column")
                if not isinstance(column, str) or not _exists(column) or column in protected:
                    return False
                strategy = str(params.get("strategy") or "").lower()
                col_type = _column_type(column)
                if col_type == "numeric":
                    return strategy in {"constant", "mean", "median", "mode"}
                return strategy in {"constant", "mode"}
            if tool_name == "clip_outliers":
                column = params.get("column")
                return (
                    isinstance(column, str)
                    and _exists(column)
                    and column not in protected
                    and _column_type(column) == "numeric"
                )

        if phase == "indepth_eda":
            if tool_name == "segment_analysis":
                cols = params.get("feature_columns")
                if not isinstance(cols, list) or not cols:
                    return False
                return _all_exist(cols) and not any(col in protected for col in cols)
            return True

        if phase == "feature_engineering":
            if tool_name == "extract_regex":
                column = params.get("column")
                new_col = params.get("new_col")
                return (
                    isinstance(column, str)
                    and isinstance(new_col, str)
                    and _exists(column)
                    and column not in protected
                    and new_col not in protected
                    and _column_type(column) in string_like
                )
            if tool_name == "create_binning":
                column = params.get("column")
                new_col = params.get("new_col")
                return (
                    isinstance(column, str)
                    and isinstance(new_col, str)
                    and _exists(column)
                    and column not in protected
                    and new_col not in protected
                    and _column_type(column) == "numeric"
                )
            if tool_name == "create_interaction":
                cols = params.get("cols")
                new_col = params.get("new_col")
                if not isinstance(cols, list) or not cols or not isinstance(new_col, str):
                    return False
                return (
                    _all_exist(cols)
                    and not any(col in protected for col in cols)
                    and new_col not in protected
                    and all(_column_type(col) == "numeric" for col in cols)
                )
            if tool_name in {"group_rare_categories", "encode_categorical"}:
                column = params.get("column")
                return (
                    isinstance(column, str)
                    and _exists(column)
                    and column not in protected
                    and _column_type(column) in string_like
                )

        if phase == "model_building" and tool_name == "configure_candidate_models":
            models = params.get("models")
            if not isinstance(models, list) or not models:
                return False
            if len(models) > 5:
                return False
            prediction_kind = str(
                target_spec_summary.get("prediction_kind") or ""
            ).lower()
            if normalized_task_type == "clustering":
                allowed_model_names = {"KMeans", "DBSCAN"}
            elif normalized_task_type == "multi_output":
                allowed_model_names = {
                    "Ridge",
                    "RandomForest",
                    "ExtraTrees",
                    "GradientBoosting",
                    "LightGBM",
                    "XGBoost",
                }
            elif normalized_task_type == "regression":
                allowed_model_names = {
                    "Ridge",
                    "Lasso",
                    "RandomForest",
                    "ExtraTrees",
                    "GradientBoosting",
                    "LightGBM",
                    "XGBoost",
                }
            elif normalized_task_type == "multilabel":
                allowed_model_names = {
                    "LogisticRegression",
                    "RandomForest",
                    "ExtraTrees",
                    "LightGBM",
                    "XGBoost",
                }
            else:
                allowed_model_names = {
                    "LogisticRegression",
                    "RandomForest",
                    "ExtraTrees",
                    "GradientBoosting",
                    "HistGradientBoosting",
                    "LightGBM",
                    "SVC",
                    "XGBoost",
                }
                if prediction_kind == "probability":
                    allowed_model_names.discard("SVC")
            return all(
                isinstance(entry, dict)
                and str(entry.get("name") or "").strip() in allowed_model_names
                for entry in models
            )

        return True

    @staticmethod
    def _normalize_model_building_parameters(
        *,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(params)

        if tool_name == "configure_candidate_models":
            models = normalized.get("models")
            if isinstance(models, list):
                cleaned_models: list[dict[str, Any]] = []
                name_aliases = {
                    "random_forest": "RandomForest",
                    "randomforest": "RandomForest",
                    "extra_trees": "ExtraTrees",
                    "extratrees": "ExtraTrees",
                    "extra_trees_classifier": "ExtraTrees",
                    "extra_trees_regressor": "ExtraTrees",
                    "logistic_regression": "LogisticRegression",
                    "logisticregression": "LogisticRegression",
                    "gradient_boosting": "HistGradientBoosting",
                    "gradientboosting": "HistGradientBoosting",
                    "hist_gradient_boosting": "HistGradientBoosting",
                    "histgradientboosting": "HistGradientBoosting",
                    "ridge_regression": "Ridge",
                    "svc": "SVC",
                    "lightgbm": "LightGBM",
                    "lgbm": "LightGBM",
                    "lgb": "LightGBM",
                    "xgboost": "XGBoost",
                    "kmeans": "KMeans",
                    "dbscan": "DBSCAN",
                }
                seen_names: set[str] = set()
                for entry in models:
                    if isinstance(entry, str) and entry.strip():
                        raw_name = entry.strip()
                        name = name_aliases.get(raw_name.lower(), raw_name)
                        if name in seen_names:
                            continue
                        cleaned_models.append({"name": name})
                        seen_names.add(name)
                    elif isinstance(entry, dict):
                        raw_name = str(entry.get("name") or "").strip()
                        name = name_aliases.get(raw_name.lower(), raw_name)
                        if not name or name in seen_names:
                            continue
                        item = {"name": name}
                        if isinstance(entry.get("param_grid"), dict):
                            item["param_grid"] = entry["param_grid"]
                        cleaned_models.append(item)
                        seen_names.add(name)
                normalized["models"] = cleaned_models[:5]

        return normalized

    def _phase_tool_allowlist(self, normalized_task_type: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for phase in self.supported_phases:
            phase_cfg = PHASE_TOOL_ALLOWLISTS.get(phase, {})
            tools = phase_cfg.get(normalized_task_type) or phase_cfg.get("default") or []
            result[phase] = list(tools)
        return result

    def _execution_plan_from_agent_plan(
        self,
        target_phase: str,
        phase_tool_allowlist: dict[str, list[str]],
        agent_plan: ToolCallPlan,
    ) -> ExecutionPlan:
        success_criteria = self._build_success_criteria_bundle_from_agent_plan(
            target_phase=target_phase,
            phase_tool_allowlist=phase_tool_allowlist,
            agent_plan=agent_plan,
        )

        return ExecutionPlan(
            phases=list(self.supported_phases),
            phase_objectives=dict(DEFAULT_PHASE_OBJECTIVES),
            success_criteria=success_criteria,
            review_points=list(DEFAULT_REVIEW_POINTS),
            fallback_rules=list(DEFAULT_FALLBACK_RULES),
        )

    def _build_success_criteria_bundle_from_agent_plan(
        self,
        *,
        target_phase: str,
        phase_tool_allowlist: dict[str, list[str]],
        agent_plan: ToolCallPlan,
    ) -> dict[str, Any]:
        bundled: dict[str, Any] = {}
        for phase in self.supported_phases:
            bundled[phase] = {
                "criteria": list(DEFAULT_SUCCESS_CRITERIA[phase]),
                "allowed_tools": list(phase_tool_allowlist.get(phase, [])),
                "agent_plan": {
                    "plan_name": f"{phase}_pending_plan",
                    "steps": [],
                },
            }

        bundled[target_phase] = {
            "criteria": list(DEFAULT_SUCCESS_CRITERIA[target_phase]),
            "allowed_tools": list(phase_tool_allowlist.get(target_phase, [])),
            "agent_plan": agent_plan.model_dump(mode="python"),
        }
        return bundled

    async def _plan_current_phase_from_state(
        self,
        *,
        state: GraphState,
        target_phase: str,
        normalized_task_type: str,
    ) -> ToolCallPlan:
        target_spec_summary = self._target_spec_summary(state)

        if target_phase == "preliminary_eda":
            return ToolCallPlan(
                plan_name="preliminary_eda_observability_plan",
                steps=[],
            )

        profile_summary = self._phase_profile_summary(
            state=state,
            target_phase=target_phase,
            target_spec_summary=target_spec_summary,
        )

        if target_phase in DETERMINISTIC_ONLY_PLANNER_PHASES:
            return self._ensure_nonempty_phase_plan(
                phase=target_phase,
                normalized_task_type=normalized_task_type,
                plan=ToolCallPlan(
                    plan_name=f"{target_phase}_{normalized_task_type}_noop_plan",
                    steps=[],
                ),
                profile_summary=profile_summary,
                target_spec_summary=target_spec_summary,
            )

        if target_phase == "data_cleaning":
            llm_plan = await self.plan_data_cleaning(profile_summary, target_spec_summary)
        elif target_phase == "indepth_eda":
            llm_plan = await self.plan_indepth_eda(profile_summary, target_spec_summary)
        elif target_phase == "feature_engineering":
            llm_plan = await self.plan_feature_engineering(
                profile_summary,
                target_spec_summary,
            )
        elif target_phase == "model_building":
            llm_plan = await self.plan_model_building(profile_summary, target_spec_summary)
        else:
            llm_plan = ToolCallPlan(
                plan_name=f"{target_phase}_{normalized_task_type}_noop_plan",
                steps=[],
            )

        return self._ensure_nonempty_phase_plan(
            phase=target_phase,
            normalized_task_type=normalized_task_type,
            plan=llm_plan,
            profile_summary=profile_summary,
            target_spec_summary=target_spec_summary,
        )

    def _retry_context_for_phase(
        self,
        *,
        phase: str,
        profile_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        review_state = profile_summary.get("review_state") or {}
        latest_status = str(review_state.get("latest_review_status") or "").strip()
        pending_phase = str(review_state.get("pending_review_phase") or "").strip()
        if latest_status != "retry" or pending_phase != phase:
            return None

        review_history = list(review_state.get("review_history") or [])
        latest_record = next(
            (
                record
                for record in reversed(review_history)
                if isinstance(record, dict)
                and str(record.get("phase") or "").strip() == phase
                and str(record.get("status") or "").strip() == "retry"
            ),
            None,
        )
        if not isinstance(latest_record, dict):
            return {
                "attempt": int(review_state.get("retry_count_by_phase", {}).get(phase) or 0),
                "last_status": latest_status,
                "blocking_failures": [],
                "warnings": [],
                "previous_plan_name": "",
                "previous_plan_steps": [],
            }

        previous_plan_name = ""
        previous_plan_steps: list[dict[str, Any]] = []
        phase_plan_records = dict(profile_summary.get("phase_plan_records") or {})
        phase_plan_path = phase_plan_records.get(phase)
        if phase_plan_path:
            try:
                plan_payload = json.loads(Path(str(phase_plan_path)).read_text(encoding="utf-8"))
                previous_plan_name = str(plan_payload.get("plan_name") or "").strip()
                raw_steps = list(plan_payload.get("steps") or [])
                for step in raw_steps[:5]:
                    if not isinstance(step, dict):
                        continue
                    previous_plan_steps.append(
                        {
                            "tool": str(step.get("tool") or "").strip(),
                            "parameters": dict(step.get("parameters") or {}),
                        }
                    )
            except Exception:
                previous_plan_name = ""
                previous_plan_steps = []

        return {
            "attempt": int(latest_record.get("attempt") or 0),
            "last_status": latest_status,
            "blocking_failures": list(latest_record.get("blocking_failures") or [])[:3],
            "warnings": list(latest_record.get("warnings") or [])[:3],
            "previous_plan_name": previous_plan_name,
            "previous_plan_steps": previous_plan_steps,
        }

    def _target_spec_summary(self, state: GraphState) -> dict[str, Any]:
        return {
            "status": state.target_spec.status,
            "target_columns": list(state.target_spec.target_columns),
            "target_count": len(state.target_spec.target_columns),
            "id_columns": list(state.target_spec.submission.id_columns or []),
            "target_mode": state.target_spec.target_mode,
            "task_family": state.target_spec.task_family,
            "prediction_kind": state.target_spec.prediction_kind,
            "candidate_metrics": list(state.target_spec.evaluation.candidate_metrics or []),
            "primary_metric": state.target_spec.evaluation.primary_metric,
            "submission_columns": list(state.target_spec.submission.submission_columns or []),
        }

    def _phase_profile_summary(
        self,
        *,
        state: GraphState,
        target_phase: str,
        target_spec_summary: dict[str, Any],
    ) -> dict[str, Any]:
        target_cols = list(target_spec_summary.get("target_columns") or [])
        train_path = self._phase_train_path(state, target_phase)
        test_path = self._phase_test_path(state, target_phase)

        summary: dict[str, Any] = {
            "target_spec_summary": target_spec_summary,
            "phase": target_phase,
            "eda_findings": state.data_state.eda_findings,
            "schema_summary": state.data_state.schema_summary,
            "cleaning_summary": state.data_state.cleaning_summary,
            "indepth_eda_summary": state.data_state.indepth_eda_summary,
            "selected_features": list(state.feature_state.selected_features or []),
            "review_state": {
                "latest_review_status": state.review_state.latest_review_status,
                "pending_review_phase": state.review_state.pending_review_phase,
                "retry_count_by_phase": dict(state.review_state.retry_count_by_phase or {}),
                "review_history": list(state.review_state.review_history or []),
            },
            "phase_plan_records": dict(state.artifacts.get("phase_plan_records") or {}),
        }

        if train_path and Path(train_path).exists():
            train_df = pd.read_csv(train_path)
            summary.update(profile_dataset(train_df, target_cols=target_cols))
            summary["train_path"] = str(train_path)
        if test_path and Path(test_path).exists():
            test_df = pd.read_csv(test_path)
            summary["test_dataset_shape"] = {
                "rows": int(test_df.shape[0]),
                "columns": int(test_df.shape[1]),
            }
        return summary

    def _phase_train_path(self, state: GraphState, target_phase: str) -> str:
        if target_phase == "model_building":
            return (
                state.feature_state.engineered_train_path
                or state.data_state.current_train_path
            )
        return state.data_state.current_train_path

    def _phase_test_path(self, state: GraphState, target_phase: str) -> str:
        if target_phase == "model_building":
            return (
                state.feature_state.engineered_test_path
                or state.data_state.current_test_path
            )
        return state.data_state.current_test_path

    def _ensure_nonempty_phase_plan(
        self,
        *,
        phase: str,
        normalized_task_type: str,
        plan: ToolCallPlan,
        profile_summary: dict[str, Any],
        target_spec_summary: dict[str, Any],
    ) -> ToolCallPlan:
        if plan.steps:
            return plan

        fallback_steps = self._deterministic_phase_steps(
            phase=phase,
            normalized_task_type=normalized_task_type,
            profile_summary=profile_summary,
            target_spec_summary=target_spec_summary,
        )
        if phase == "data_cleaning" and not fallback_steps:
            return ToolCallPlan(
                plan_name="data_cleaning_no_action_needed",
                steps=[],
            )
        return ToolCallPlan(
            plan_name=f"{phase}_{normalized_task_type}_deterministic_default",
            steps=fallback_steps,
        )

    def _deterministic_phase_steps(
        self,
        *,
        phase: str,
        normalized_task_type: str,
        profile_summary: dict[str, Any],
        target_spec_summary: dict[str, Any],
    ) -> list[ToolCall]:
        columns = profile_summary.get("columns") or {}
        if not isinstance(columns, dict):
            columns = {}
        target_cols = set(target_spec_summary.get("target_columns") or [])
        id_cols = set(target_spec_summary.get("id_columns") or [])
        protected = target_cols | id_cols

        def _feature_items() -> list[tuple[str, dict[str, Any]]]:
            items: list[tuple[str, dict[str, Any]]] = []
            for name, meta in columns.items():
                if name in protected or not isinstance(meta, dict):
                    continue
                items.append((name, meta))
            return items

        feature_items = _feature_items()

        if phase == "data_cleaning":
            steps: list[ToolCall] = []
            drop_candidates: list[str] = []
            missing_candidates: list[tuple[float, str, dict[str, Any]]] = []
            for name, meta in feature_items:
                missing_rate = float(meta.get("missing_rate") or 0.0)
                if missing_rate >= 0.98:
                    drop_candidates.append(name)
                elif missing_rate > 0:
                    missing_candidates.append((missing_rate, name, meta))

            if drop_candidates:
                steps.append(
                    ToolCall(
                        tool="drop_columns",
                        parameters={"columns": drop_candidates[:5]},
                        reasoning="Drop non-target columns that are almost entirely missing.",
                    )
                )

            for _, name, meta in sorted(missing_candidates, reverse=True)[:6]:
                strategy = "median" if meta.get("type") == "numeric" else "mode"
                steps.append(
                    ToolCall(
                        tool="fill_missing",
                        parameters={"column": name, "strategy": strategy},
                        reasoning="Impute observed missing values with a conservative default strategy.",
                    )
                )
            return steps

        if phase == "indepth_eda":
            steps = [
                ToolCall(
                    tool="target_analysis",
                    parameters={},
                    reasoning="Summarize target distributions before deeper analysis.",
                ),
                ToolCall(
                    tool="feature_feature_relationships",
                    parameters={},
                    reasoning="Inspect relationships among non-target features.",
                ),
                ToolCall(
                    tool="outlier_analysis",
                    parameters={},
                    reasoning="Inspect numeric outliers that may affect modeling.",
                ),
            ]
            first_segment_col = next(
                (
                    name
                    for name, meta in feature_items
                    if meta.get("type") == "categorical"
                    and 1 < int(meta.get("nunique") or 0) <= 20
                ),
                None,
            )
            if first_segment_col:
                steps.append(
                    ToolCall(
                        tool="segment_analysis",
                        parameters={"feature_columns": [first_segment_col]},
                        reasoning="Compare target behavior across a representative categorical segment.",
                    )
                )
            return steps

        if phase == "feature_engineering":
            steps = []
            first_numeric = next(
                (
                    name
                    for name, meta in feature_items
                    if meta.get("type") == "numeric"
                    and int(meta.get("nunique") or 0) >= 10
                ),
                None,
            )
            if first_numeric:
                steps.append(
                    ToolCall(
                        tool="create_binning",
                        parameters={
                            "column": first_numeric,
                            "n_bins": 5,
                            "new_col": f"{first_numeric}_bin",
                            "strategy": "quantile",
                        },
                        reasoning="Create a robust bucketed version of a high-variance numeric feature.",
                    )
                )
            first_high_card_cat = next(
                (
                    name
                    for name, meta in feature_items
                    if meta.get("type") == "categorical"
                    and int(meta.get("nunique") or 0) >= 8
                ),
                None,
            )
            if first_high_card_cat:
                steps.append(
                    ToolCall(
                        tool="group_rare_categories",
                        parameters={
                            "column": first_high_card_cat,
                            "threshold": 0.02,
                            "rare_label": "Other",
                        },
                        reasoning="Reduce fragmentation in a higher-cardinality categorical feature.",
                    )
                )
            return steps

        if phase == "model_building":
            task_family = str(target_spec_summary.get("task_family") or "")
            primary_metric = str(target_spec_summary.get("primary_metric") or "")
            model_names = self._default_model_names(
                normalized_task_type=normalized_task_type,
                task_family=task_family,
                prediction_kind=str(target_spec_summary.get("prediction_kind") or ""),
            )
            return [
                ToolCall(
                    tool="configure_preprocessing",
                    parameters={
                        "numeric_impute": "median",
                        "scaler": "standard",
                        "cat_encode": "onehot",
                        "cat_impute": "most_frequent",
                        "onehot_max_cats": 25,
                    },
                    reasoning="Use a conservative preprocessing baseline that supports mixed tabular data.",
                ),
                ToolCall(
                    tool="configure_scoring",
                    parameters={
                        "primary_metric": primary_metric,
                    },
                    reasoning="Align scoring with the resolved target contract.",
                ),
                ToolCall(
                    tool="configure_candidate_models",
                    parameters={
                        "models": [{"name": name} for name in model_names],
                    },
                    reasoning="Evaluate a compact, robust model family suited to the task type.",
                ),
            ]

        return []

    @staticmethod
    def _default_model_names(
        *,
        normalized_task_type: str,
        task_family: str,
        prediction_kind: str,
    ) -> list[str]:
        prediction_kind = prediction_kind.lower()
        if normalized_task_type == "clustering":
            return ["KMeans", "DBSCAN"]
        if normalized_task_type == "multilabel":
            return ["LogisticRegression", "RandomForest", "ExtraTrees", "LightGBM", "XGBoost"]
        if normalized_task_type == "multi_output":
            return ["Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "LightGBM"]
        if "classif" in task_family:
            if prediction_kind == "probability":
                return ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting", "LightGBM"]
            return ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting", "LightGBM"]
        return ["Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "LightGBM"]

    def _validated_agent_plan(
        self,
        raw_plan: Any,
        target_phase: str,
        allowed_tools: list[str],
        normalized_task_type: str,
    ) -> dict[str, Any]:
        fallback = {
            "plan_name": f"{target_phase}_{normalized_task_type}_noop_plan",
            "steps": [],
        }
        if not isinstance(raw_plan, dict):
            return fallback

        plan_name = str(raw_plan.get("plan_name") or fallback["plan_name"]).strip()
        raw_steps = raw_plan.get("steps")
        if not isinstance(raw_steps, list):
            return {"plan_name": plan_name, "steps": []}

        cleaned_steps: list[dict[str, Any]] = []
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            tool_name = str(step.get("tool", "")).strip()
            if not tool_name or tool_name not in allowed_tools:
                continue
            parameters = step.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            reasoning = str(step.get("reasoning", "")).strip()
            cleaned_steps.append(
                {
                    "tool": tool_name,
                    "parameters": parameters,
                    "reasoning": reasoning,
                }
            )

        return {"plan_name": plan_name, "steps": cleaned_steps}

    def _list_of_strings(self, value: Any, default: list[str]) -> list[str]:
        if not isinstance(value, list):
            return list(default)
        cleaned = [str(item) for item in value if str(item).strip()]
        return cleaned or list(default)


async def run_planner(state: GraphState) -> dict:
    """Execute the planner phase and return partial state updates."""

    return await PlannerAgent().run(state)


__all__ = ["PlannerAgent", "EXECUTION_PHASES", "run_planner"]
