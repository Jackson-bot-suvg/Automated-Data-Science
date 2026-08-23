"""Technical summarizer agent for final workflow reporting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.schemas.state_schema import GraphState
from app.tools.run_manifest import write_run_manifest
from app.utils.target_scope import get_id_columns, get_target_columns


class TechnicalSummarizerAgent:
    """Generate the final technical report after model building is approved."""

    final_execution_phase = "model_building"
    summary_phase = "technical_summary"

    async def run(self, state: GraphState) -> dict:
        """Run technical summarization or no-op until the final phase is approved."""

        if not self._ready_for_final_summary(state):
            return {
                "messages": [
                    {
                        "role": "system",
                        "name": "technical_summarizer",
                        "content": (
                            "Skipped technical summary because "
                            f"current_phase={state.current_phase!r}."
                        ),
                    }
                ]
            }

        summary_context = self._summary_context(state)
        summary_payload = self._build_summary_payload(summary_context)
        report_path = self._technical_report_path(state)

        fatal_errors = list(state.review_state.fatal_errors)
        try:
            self._write_text(report_path, summary_payload["report_markdown"])
        except Exception as exc:  # noqa: BLE001
            error_msg = (
                f"[summarizer] Failed to write technical report to {report_path}: {exc}"
            )
            fatal_errors.append(error_msg)
            return {
                "current_phase": self.summary_phase,
                "phase_history": [self.summary_phase],
                "review_state": {
                    **state.review_state.model_dump(),
                    "fatal_errors": fatal_errors,
                },
                "messages": [
                    {
                        "role": "system",
                        "name": "technical_summarizer",
                        "content": error_msg,
                    }
                ],
            }

        manifest_path = str(state.artifacts.get("run_manifest") or "")
        if manifest_path:
            write_run_manifest(
                workspace_root=state.runtime_state.workspace_root,
                manifest_path=manifest_path,
                run_id=state.runtime_state.run_id or state.task_config.task_id,
                thread_id=state.runtime_state.thread_id,
                current_phase=self.summary_phase,
                phase_history=list(state.phase_history or []) + [self.summary_phase],
                task_config=state.task_config.model_dump(mode="python"),
                target_spec=state.target_spec.model_dump(mode="python"),
                artifacts={
                    **state.artifacts,
                    "technical_report": str(report_path),
                },
                runtime_state=state.runtime_state.model_dump(mode="python"),
                data_state=state.data_state.model_dump(mode="python"),
                feature_state=state.feature_state.model_dump(mode="python"),
                model_state=state.model_state.model_dump(mode="python"),
                review_state=state.review_state.model_dump(mode="python"),
                report_state={
                    **state.report_state.model_dump(mode="python"),
                    "technical_report_path": str(report_path),
                    "limitations": summary_payload["limitations"],
                    "executive_summary": summary_payload["executive_summary"],
                },
            )

        return {
            "current_phase": self.summary_phase,
            "phase_history": [self.summary_phase],
            "report_state": {
                **state.report_state.model_dump(mode="python"),
                "technical_report_path": str(report_path),
                "executive_summary": summary_payload["executive_summary"],
                "limitations": summary_payload["limitations"],
            },
            "artifacts": {
                **state.artifacts,
                "technical_report": str(report_path),
                "run_manifest": manifest_path or state.artifacts.get("run_manifest", ""),
            },
            "messages": [
                {
                    "role": "system",
                    "name": "technical_summarizer",
                    "content": "Generated final technical report.",
                }
            ],
        }

    def _ready_for_final_summary(self, state: GraphState) -> bool:
        return (
            state.current_phase == self.final_execution_phase
            and state.review_state.latest_review_status == "approved"
        )

    def _technical_report_path(self, state: GraphState) -> Path:
        workspace_root = Path(state.runtime_state.workspace_root or ".").resolve()
        artifact_path = Path(
            str(
                state.artifacts.get("technical_report")
                or "artifacts/reports/technical_report.md"
            )
        )
        if not artifact_path.is_absolute():
            artifact_path = workspace_root / artifact_path
        return artifact_path.resolve()

    def _summary_context(self, state: GraphState) -> dict[str, Any]:
        return {
            "workspace_root": str(
                Path(state.runtime_state.workspace_root or ".").resolve()
            ),
            "task_config": state.task_config.model_dump(mode="python"),
            "problem_understanding": state.problem_understanding.model_dump(
                mode="python"
            ),
            "target_spec": state.target_spec.model_dump(mode="python"),
            "target_contract": self._resolve_target_contract(state),
            "data_state": state.data_state.model_dump(mode="python"),
            "feature_state": state.feature_state.model_dump(mode="python"),
            "model_state": state.model_state.model_dump(mode="python"),
            "artifacts": {
                key: value for key, value in state.artifacts.items() if key != "figures"
            },
            "figures": list(state.artifacts.get("figures", [])),
        }

    def _resolve_target_contract(self, state: GraphState) -> dict[str, Any]:
        target_spec = state.target_spec
        model_meta = state.model_state.validation_metrics.get("_meta", {})

        target_columns = self._clean_strings(target_spec.target_columns)
        used_legacy_target_fallback = False
        if not target_columns:
            target_columns = self._clean_strings(get_target_columns(state))
            used_legacy_target_fallback = bool(target_columns)

        id_columns = self._clean_strings(target_spec.submission.id_columns)
        used_legacy_id_fallback = False
        if not id_columns:
            id_columns = self._clean_strings(get_id_columns(state))
            used_legacy_id_fallback = bool(id_columns)

        output_columns = self._clean_strings(target_spec.submission.output_columns)
        if not output_columns:
            output_columns = list(target_columns)

        submission_columns = self._clean_strings(
            target_spec.submission.submission_columns
        )
        if not submission_columns and (id_columns or output_columns):
            submission_columns = [*id_columns, *output_columns]

        raw_task_family = (
            str(target_spec.task_family or "").strip()
            or str(model_meta.get("task_family") or "").strip()
            or str(state.problem_understanding.predicted_task_type or "").strip()
            or str(state.task_config.task_type or "").strip()
        )
        raw_target_mode = (
            str(target_spec.target_mode or "").strip()
            or str(model_meta.get("target_mode") or "").strip()
        )
        task_family = self._canonical_task_family(
            raw_task_family=raw_task_family,
            raw_target_mode=raw_target_mode,
            target_columns=target_columns,
        )
        target_mode = self._canonical_target_mode(
            raw_mode=raw_target_mode,
            task_family=task_family,
            target_columns=target_columns,
        )

        candidate_metrics = self._clean_strings(
            target_spec.evaluation.candidate_metrics
            or state.problem_understanding.candidate_metrics
        )
        primary_metric = (
            str(target_spec.evaluation.primary_metric or "").strip()
            or str(target_spec.evaluation.metric_hint or "").strip()
            or str(model_meta.get("primary_metric") or "").strip()
            or str(state.task_config.metric_hint or "").strip()
            or self._guess_primary_metric(
                candidate_metrics=candidate_metrics,
                validation_metrics=state.model_state.validation_metrics,
            )
        )
        prediction_kind = (
            str(target_spec.prediction_kind or "").strip()
            or str(model_meta.get("prediction_kind") or "").strip()
            or self._infer_prediction_kind(
                task_family=task_family,
                target_mode=target_mode,
                primary_metric=primary_metric,
            )
        )

        return {
            "status": target_spec.status,
            "target_columns": target_columns,
            "id_columns": id_columns,
            "output_columns": output_columns,
            "submission_columns": submission_columns,
            "task_family": task_family,
            "target_mode": target_mode,
            "prediction_kind": prediction_kind,
            "primary_metric": primary_metric,
            "candidate_metrics": candidate_metrics,
            "sample_submission_path": str(
                target_spec.submission.sample_submission_path or ""
            ).strip(),
            "used_legacy_fallback": (
                used_legacy_target_fallback or used_legacy_id_fallback
            ),
        }

    def _build_summary_payload(self, summary_context: dict[str, Any]) -> dict[str, Any]:
        executive_summary = self._build_executive_summary(summary_context)
        limitations = self._derive_limitations(summary_context)
        return {
            "executive_summary": executive_summary,
            "limitations": limitations,
            "report_markdown": self._render_report(
                summary_context=summary_context,
                executive_summary=executive_summary,
                limitations=limitations,
            ),
        }

    def _build_executive_summary(self, summary_context: dict[str, Any]) -> str:
        target_contract = summary_context["target_contract"]
        model_state = summary_context["model_state"]
        best_model = model_state.get("best_model_name") or "not selected"
        primary_metric = target_contract.get("primary_metric") or "primary_metric"
        best_metrics = self._best_model_metrics(model_state)
        primary_value = self._metric_value(
            best_metrics or (model_state.get("validation_metrics") or {}),
            primary_metric,
        )
        task_family = target_contract.get("task_family") or "unknown_task"
        target_label = self._target_label(target_contract.get("target_columns") or [])

        metric_fragment = (
            f"{primary_metric}={primary_value}" if primary_value else primary_metric
        )
        return (
            f"This workflow trained a {task_family} solution for {target_label}. "
            f"The selected model is {best_model}, evaluated primarily with {metric_fragment}."
        )

    def _derive_limitations(self, summary_context: dict[str, Any]) -> list[str]:
        target_contract = summary_context["target_contract"]
        data_state = summary_context["data_state"]
        feature_state = summary_context["feature_state"]
        model_state = summary_context["model_state"]
        limitations: list[str] = []

        if target_contract.get("used_legacy_fallback"):
            limitations.append(
                "The target contract required compatibility fallback fields because target_spec was incomplete."
            )
        if not model_state.get("validation_metrics"):
            limitations.append(
                "Validation metrics are missing, so model quality cannot be fully audited from state alone."
            )
        if not model_state.get("prediction_path"):
            limitations.append(
                "Prediction output path is missing, so the final scoring artifact cannot be traced from the report."
            )
        if not feature_state.get("feature_importance_summary"):
            limitations.append(
                "Feature importance is unavailable, so driver-level interpretation is limited."
            )
        if not feature_state.get("selected_features"):
            limitations.append(
                "Selected feature metadata is unavailable, so the engineered feature set is only partially documented."
            )
        if not data_state.get("cleaning_summary"):
            limitations.append(
                "Cleaning operations were not recorded in a structured summary."
            )
        if (
            len(target_contract.get("target_columns") or []) > 1
            and not self._has_per_target_signal(model_state.get("validation_metrics") or {})
        ):
            limitations.append(
                "The report does not contain a per-target metric breakdown for the multi-target task."
            )

        deduped = self._dedupe_preserve_order(limitations)
        if not deduped:
            return [
                "No major technical limitations were recorded in structured state, but external validation and deployment checks are still recommended."
            ]
        return deduped[:6]

    def _render_report(
        self,
        *,
        summary_context: dict[str, Any],
        executive_summary: str,
        limitations: list[str],
    ) -> str:
        target_contract = summary_context["target_contract"]
        problem = summary_context["problem_understanding"]
        task_config = summary_context["task_config"]
        artifacts = summary_context["artifacts"]
        model_state = summary_context["model_state"]

        report_lines = [
            "# Technical Report",
            "",
            "## Executive Summary",
            executive_summary,
            "",
            "## Problem Definition",
            f"- Task summary: {self._problem_summary(problem, task_config)}",
            f"- Task family: `{target_contract.get('task_family') or 'unknown'}`",
            f"- Target mode: `{target_contract.get('target_mode') or 'unknown'}`",
            f"- Target columns: {self._backticked_list(target_contract.get('target_columns') or [])}",
            f"- Prediction kind: `{target_contract.get('prediction_kind') or 'unknown'}`",
            f"- Primary metric: `{target_contract.get('primary_metric') or 'unknown'}`",
            f"- Candidate metrics: {self._backticked_list(target_contract.get('candidate_metrics') or [])}",
            "",
            "## Dataset Snapshot",
            *self._dataset_snapshot_lines(summary_context),
            "",
            "## Data Preparation Summary",
            *self._data_preparation_lines(summary_context),
            "",
            "## Feature Engineering Summary",
            *self._feature_engineering_lines(summary_context),
            "",
            "## Modeling Summary",
            *self._modeling_lines(summary_context),
            "",
            "## Prediction Outputs",
            *self._prediction_output_lines(summary_context),
            "",
            "## Business Translation Inputs",
            *self._business_input_lines(summary_context, limitations),
            "",
            "## Artifacts",
            *self._artifact_lines(summary_context),
            "",
            "## Limitations",
            *[f"- {item}" for item in limitations],
            "",
        ]
        return "\n".join(report_lines)

    def _problem_summary(
        self, problem: dict[str, Any], task_config: dict[str, Any]
    ) -> str:
        summary = self._english_text(
            problem.get("task_summary"),
            default="A predictive modeling workflow was executed on the provided dataset.",
        )
        raw_task = self._english_text(task_config.get("raw_task"), default="")
        if raw_task and raw_task.casefold() != summary.casefold():
            return f"{summary} User task: {raw_task}"
        return summary

    def _dataset_snapshot_lines(self, summary_context: dict[str, Any]) -> list[str]:
        target_contract = summary_context["target_contract"]
        problem = summary_context["problem_understanding"]
        data_state = summary_context["data_state"]

        lines = [
            f"- Training shape: {self._shape_text(data_state.get('train_shape'))}",
            f"- Test shape: {self._shape_text(data_state.get('test_shape'))}",
        ]

        data_overview = problem.get("data_overview") or {}
        if data_overview.get("n_rows") or data_overview.get("n_cols"):
            lines.append(
                f"- Raw dataset overview: {data_overview.get('n_rows') or '?'} rows and {data_overview.get('n_cols') or '?'} columns."
            )

        column_types = data_overview.get("column_types") or {}
        if column_types:
            lines.append(
                f"- Column type coverage: {len(column_types)} columns profiled."
            )

        missing_summary = data_overview.get("missingness_summary") or {}
        if isinstance(missing_summary, dict) and missing_summary:
            ranked_missing = sorted(
                (
                    (str(key), value)
                    for key, value in missing_summary.items()
                    if isinstance(value, (int, float))
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if ranked_missing:
                top_missing = ", ".join(
                    f"`{name}` ({self._format_number(value)})"
                    for name, value in ranked_missing[:3]
                )
                lines.append(f"- Most visible missingness: {top_missing}.")

        target_columns = target_contract.get("target_columns") or []
        id_columns = target_contract.get("id_columns") or []
        if column_types:
            feature_count = max(len(column_types) - len(target_columns) - len(id_columns), 0)
            lines.append(f"- Approximate raw feature count: {feature_count}.")

        return lines

    def _data_preparation_lines(self, summary_context: dict[str, Any]) -> list[str]:
        data_state = summary_context["data_state"]
        lines: list[str] = []

        cleaning_summary = data_state.get("cleaning_summary") or {}
        cleaning_steps = self._operation_summaries(cleaning_summary, limit=5)
        if cleaning_steps:
            lines.extend(f"- {item}" for item in cleaning_steps)
        else:
            lines.append("- No structured cleaning operations were recorded.")

        quality_flags = self._english_list(data_state.get("data_quality_flags") or [])
        if quality_flags:
            lines.append(
                f"- Data quality notes: {self._preview_text_list(quality_flags, limit=3)}"
            )

        eda_findings = data_state.get("eda_findings") or {}
        eda_highlights = self._finding_highlights(eda_findings, limit=3)
        if eda_highlights:
            lines.append(
                f"- EDA highlights: {self._preview_text_list(eda_highlights, limit=3)}"
            )

        indepth_summary = data_state.get("indepth_eda_summary") or {}
        indepth_highlights = self._finding_highlights(indepth_summary, limit=3)
        if indepth_highlights:
            lines.append(
                f"- In-depth EDA highlights: {self._preview_text_list(indepth_highlights, limit=3)}"
            )

        return lines

    def _feature_engineering_lines(self, summary_context: dict[str, Any]) -> list[str]:
        feature_state = summary_context["feature_state"]
        selected_features = feature_state.get("selected_features") or []
        dropped_features = feature_state.get("dropped_features") or []

        lines = [
            f"- Selected features: {len(selected_features)}",
            f"- Dropped features: {len(dropped_features)}",
        ]

        if selected_features:
            lines.append(
                f"- Feature set preview: {self._backticked_list(selected_features[:10])}"
            )

        transform_summary = feature_state.get("feature_transform_summary") or {}
        transform_steps = self._operation_summaries(transform_summary, limit=5)
        if transform_steps:
            lines.extend(f"- {item}" for item in transform_steps)

        leakage_checks = feature_state.get("leakage_checks") or []
        leakage_status = self._leakage_status(leakage_checks)
        lines.append(f"- Leakage assessment: {leakage_status}")

        feature_importance = feature_state.get("feature_importance_summary") or {}
        if feature_importance:
            lines.append(
                f"- Top technical drivers: {self._top_feature_text(feature_importance, limit=5)}"
            )

        return lines

    def _modeling_lines(self, summary_context: dict[str, Any]) -> list[str]:
        target_contract = summary_context["target_contract"]
        model_state = summary_context["model_state"]
        best_model = model_state.get("best_model_name") or "not selected"

        lines = [
            f"- Candidate models: {self._backticked_list(model_state.get('candidate_models') or [])}",
            f"- Selected model: `{best_model}`",
            f"- Primary metric focus: `{target_contract.get('primary_metric') or 'unknown'}`",
        ]

        best_metrics = self._best_model_metrics(model_state)
        if best_metrics:
            lines.append(
                f"- Best-model validation metrics: {self._inline_pairs(best_metrics, limit=6)}"
            )

        baseline_metrics = model_state.get("baseline_metrics") or {}
        if baseline_metrics:
            baseline_label = "Reference baseline metrics"
            baseline_meta = baseline_metrics.get("_meta") or {}
            if isinstance(baseline_meta, dict) and baseline_meta.get("source") == "out_of_fold_dummy_baseline":
                baseline_label = "Reference baseline metrics (out-of-fold dummy baseline)"
            lines.append(
                f"- {baseline_label}: {self._inline_pairs(baseline_metrics, limit=5, skip_keys={'_meta'})}"
            )

        validation_metrics = model_state.get("validation_metrics") or {}
        if validation_metrics and not best_metrics:
            lines.append(
                f"- Validation metrics: {self._inline_pairs(validation_metrics, limit=6, skip_keys={'_meta'})}"
            )

        best_params = model_state.get("best_params") or {}
        if best_params:
            lines.append(
                f"- Best hyperparameters: {self._inline_pairs(best_params, limit=6)}"
            )

        model_explanations = model_state.get("model_explanations") or {}
        explanation_items = self._finding_highlights(model_explanations, limit=2)
        if explanation_items:
            lines.append(
                f"- Model interpretation notes: {self._preview_text_list(explanation_items, limit=2)}"
            )

        return lines

    def _prediction_output_lines(self, summary_context: dict[str, Any]) -> list[str]:
        target_contract = summary_context["target_contract"]
        model_state = summary_context["model_state"]
        artifacts = summary_context["artifacts"]

        prediction_path = (
            model_state.get("prediction_path")
            or artifacts.get("prediction_file")
            or ""
        )

        lines = [
            f"- Prediction file: {self._safe_backticked(prediction_path)}",
            f"- Output columns: {self._backticked_list(target_contract.get('output_columns') or [])}",
            f"- Submission column order: {self._backticked_list(target_contract.get('submission_columns') or [])}",
            f"- Sample submission reference: {self._safe_backticked(target_contract.get('sample_submission_path'))}",
            f"- Prediction semantics: `{target_contract.get('prediction_kind') or 'unknown'}`",
        ]

        best_model_artifact = artifacts.get("best_model") or ""
        if best_model_artifact:
            lines.append(f"- Serialized model artifact: {self._safe_backticked(best_model_artifact)}")
        model_results_path = artifacts.get("model_results") or ""
        if model_results_path:
            lines.append(f"- Model results artifact: {self._safe_backticked(model_results_path)}")
        return lines

    def _business_input_lines(
        self, summary_context: dict[str, Any], limitations: list[str]
    ) -> list[str]:
        target_contract = summary_context["target_contract"]
        feature_state = summary_context["feature_state"]
        model_state = summary_context["model_state"]

        lines = [
            f"- Main target description: {self._target_label(target_contract.get('target_columns') or [])}",
            f"- Delivery-ready metric: `{target_contract.get('primary_metric') or 'unknown'}` with selected model `{model_state.get('best_model_name') or 'unknown'}`",
            f"- Prediction output type: `{target_contract.get('prediction_kind') or 'unknown'}`",
        ]

        feature_importance = feature_state.get("feature_importance_summary") or {}
        if feature_importance:
            lines.append(
                f"- Business-facing driver shortlist: {self._top_feature_text(feature_importance, limit=4)}"
            )

        if limitations:
            lines.append(
                f"- Caveats to carry into the business report: {self._preview_text_list(limitations, limit=2)}"
            )

        return lines

    def _artifact_lines(self, summary_context: dict[str, Any]) -> list[str]:
        workspace_root = Path(summary_context["workspace_root"])
        artifacts = summary_context["artifacts"]
        figures = summary_context.get("figures") or []

        lines: list[str] = []
        seen_paths: set[str] = set()

        for key, raw_path in sorted(artifacts.items()):
            if isinstance(raw_path, dict):
                for nested_key, nested_value in sorted(raw_path.items()):
                    path = str(nested_value or "").strip()
                    if not path or path in seen_paths:
                        continue
                    if self._path_exists(workspace_root, path):
                        seen_paths.add(path)
                        lines.append(f"- `{key}.{nested_key}`: `{path}`")
                continue
            path = str(raw_path or "").strip()
            if not path or path in seen_paths:
                continue
            if self._path_exists(workspace_root, path):
                seen_paths.add(path)
                lines.append(f"- `{key}`: `{path}`")

        for idx, raw_path in enumerate(figures, start=1):
            path = str(raw_path or "").strip()
            if not path or path in seen_paths:
                continue
            if self._path_exists(workspace_root, path):
                seen_paths.add(path)
                lines.append(f"- `figure_{idx}`: `{path}`")

        return lines or ["- (not available)"]

    def _operation_summaries(
        self, mapping: dict[str, Any], *, limit: int
    ) -> list[str]:
        summaries: list[str] = []
        for raw_name, raw_payload in list(mapping.items())[:limit]:
            step_name = self._step_name(raw_name)
            payload = raw_payload if isinstance(raw_payload, dict) else {}
            params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
            fragments = [step_name]

            column = self._safe_param(params.get("column"))
            new_col = self._safe_param(params.get("new_col"))
            strategy = self._safe_param(params.get("strategy"))
            method = self._safe_param(params.get("method"))
            operator = self._safe_param(params.get("operator"))
            columns = self._param_list(params.get("columns") or params.get("cols"))

            if column:
                fragments.append(f"on `{column}`")
            elif columns:
                fragments.append(f"on {self._backticked_list(columns)}")
            if new_col:
                fragments.append(f"creating `{new_col}`")
            if strategy:
                fragments.append(f"with strategy `{strategy}`")
            if method:
                fragments.append(f"using `{method}`")
            if operator:
                fragments.append(f"via `{operator}`")

            summaries.append(" ".join(fragments).strip())
        return summaries

    def _finding_highlights(self, mapping: dict[str, Any], *, limit: int) -> list[str]:
        findings: list[str] = []
        for key, value in self._flatten_mapping(mapping, max_depth=2)[:limit]:
            safe_key = self._english_text(key, default="")
            safe_value = self._english_text(value, default="")
            if safe_key and safe_value:
                findings.append(f"{safe_key}={safe_value}")
        return findings

    def _best_model_metrics(self, model_state: dict[str, Any]) -> dict[str, Any]:
        best_model_name = str(model_state.get("best_model_name") or "").strip()
        validation_metrics = model_state.get("validation_metrics") or {}
        if (
            best_model_name
            and isinstance(validation_metrics, dict)
            and isinstance(validation_metrics.get(best_model_name), dict)
        ):
            return dict(validation_metrics[best_model_name])
        return {}

    def _metric_value(self, metrics: dict[str, Any], metric_name: str) -> str:
        if not metric_name:
            return ""
        if isinstance(metrics, dict):
            exact = metrics.get(metric_name)
            if exact is not None and not isinstance(exact, dict):
                return self._scalar_text(exact)
            cv_mean = metrics.get(f"cv_{metric_name}_mean")
            if cv_mean is not None and not isinstance(cv_mean, dict):
                return self._scalar_text(cv_mean)
        best_model_metrics = {}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, dict) and (
                    metric_name in value
                    or f"cv_{metric_name}_mean" in value
                ):
                    best_model_metrics = value
                    break
        if best_model_metrics:
            exact = best_model_metrics.get(metric_name)
            if exact is not None:
                return self._scalar_text(exact)
            cv_mean = best_model_metrics.get(f"cv_{metric_name}_mean")
            if cv_mean is not None:
                return self._scalar_text(cv_mean)

        flattened = self._flatten_mapping(metrics, max_depth=2, skip_keys={"_meta"})
        for label, value in flattened:
            if label == metric_name or label.endswith(f".{metric_name}"):
                return value
            if label.endswith(f".cv_{metric_name}_mean"):
                return value
        return ""

    def _leakage_status(self, leakage_checks: list[Any]) -> str:
        texts = [self._safe_text(item) for item in leakage_checks if self._safe_text(item)]
        if not texts:
            return "not recorded"
        combined = " ".join(texts).casefold()
        if "warning" in combined or "fail" in combined:
            return self._preview_text_list(texts, limit=2)
        if "pass" in combined:
            return "pass"
        return self._preview_text_list(texts, limit=2)

    def _has_per_target_signal(self, metrics: dict[str, Any]) -> bool:
        for key, value in metrics.items():
            if isinstance(key, str) and ("target" in key.casefold() or "label" in key.casefold()):
                return True
            if isinstance(value, dict):
                nested_keys = " ".join(str(item) for item in value.keys()).casefold()
                if "target" in nested_keys or "label" in nested_keys:
                    return True
        return False

    def _flatten_mapping(
        self,
        mapping: Any,
        *,
        prefix: str = "",
        depth: int = 0,
        max_depth: int = 2,
        skip_keys: set[str] | None = None,
    ) -> list[tuple[str, str]]:
        skip = skip_keys or set()
        items: list[tuple[str, str]] = []

        if isinstance(mapping, dict):
            for key, value in mapping.items():
                key_text = str(key).strip()
                if not key_text or key_text in skip:
                    continue
                label = f"{prefix}.{key_text}" if prefix else key_text
                if isinstance(value, dict) and depth < max_depth:
                    items.extend(
                        self._flatten_mapping(
                            value,
                            prefix=label,
                            depth=depth + 1,
                            max_depth=max_depth,
                            skip_keys=skip_keys,
                        )
                    )
                elif isinstance(value, list):
                    if value:
                        preview = self._preview_text_list(
                            [self._scalar_text(item) or str(item) for item in value],
                            limit=4,
                        )
                        items.append((label, preview))
                else:
                    rendered = self._scalar_text(value)
                    if rendered:
                        items.append((label, rendered))
            return items

        rendered = self._scalar_text(mapping)
        if rendered:
            items.append((prefix or "value", rendered))
        return items

    def _top_feature_text(self, feature_importance: dict[str, Any], limit: int) -> str:
        numeric_pairs: list[tuple[str, float]] = []
        for key, value in feature_importance.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_pairs.append((str(key), float(value)))
        if not numeric_pairs:
            flattened = self._flatten_mapping(feature_importance, max_depth=1)
            if not flattened:
                return "(not available)"
            return "; ".join(f"{name}={value}" for name, value in flattened[:limit])

        ranked = sorted(
            numeric_pairs, key=lambda item: abs(item[1]), reverse=True
        )[:limit]
        return "; ".join(
            f"`{name}`={self._format_number(value)}" for name, value in ranked
        )

    def _step_name(self, raw_name: Any) -> str:
        text = str(raw_name or "").strip()
        if not text:
            return "recorded operation"
        lowered = text.casefold()
        if lowered.startswith("step:"):
            text = text.split(":", 1)[1].strip()
        return self._english_text(text, default="recorded operation")

    def _safe_param(self, value: Any) -> str:
        return self._english_text(value, default="")

    def _param_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return self._clean_strings(value)
        text = self._english_text(value, default="")
        if not text:
            return []
        if "," in text:
            return self._clean_strings(part.strip() for part in text.split(","))
        return [text]

    def _path_exists(self, workspace_root: Path, path: str) -> bool:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        return candidate.exists()

    def _canonical_target_mode(
        self, *, raw_mode: str, task_family: str, target_columns: list[str]
    ) -> str:
        mode = raw_mode.strip().lower()
        if mode in {"single", "single_target"}:
            return "single_target"
        if mode in {"multi_label", "multi_output"}:
            return mode
        if mode == "multi":
            if "multilabel" in task_family:
                return "multi_label"
            return "multi_output"
        if len(target_columns) <= 1:
            return "single_target" if target_columns else "unknown"
        if "multilabel" in task_family:
            return "multi_label"
        return "multi_output"

    def _canonical_task_family(
        self,
        *,
        raw_task_family: str,
        raw_target_mode: str,
        target_columns: list[str],
    ) -> str:
        task_family = raw_task_family.strip().lower().replace(" ", "_")
        if not task_family:
            if len(target_columns) > 1 and raw_target_mode in {"multi", "multi_output"}:
                return "multi_output"
            if len(target_columns) > 1 and raw_target_mode == "multi_label":
                return "multilabel_classification"
            return "unknown"
        if task_family == "classification" and len(target_columns) > 1:
            return (
                "multilabel_classification"
                if raw_target_mode in {"multi", "multi_label"}
                else "multi_target_classification"
            )
        if task_family == "regression" and len(target_columns) > 1:
            return "multi_output_regression"
        return task_family

    def _infer_prediction_kind(
        self, *, task_family: str, target_mode: str, primary_metric: str
    ) -> str:
        metric = primary_metric.lower()
        if any(token in metric for token in ("auc", "logloss", "brier", "prob")):
            return "probability"
        if "regression" in task_family or target_mode == "multi_output":
            return "continuous"
        return "label"

    def _guess_primary_metric(
        self, *, candidate_metrics: list[str], validation_metrics: dict[str, Any]
    ) -> str:
        if candidate_metrics:
            return candidate_metrics[0]
        items = self._flatten_mapping(
            validation_metrics, max_depth=1, skip_keys={"_meta"}
        )
        return items[0][0] if items else ""

    def _shape_text(self, shape: Any) -> str:
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            return f"{shape[0]} x {shape[1]}"
        return "(not available)"

    def _target_label(self, columns: list[str]) -> str:
        cleaned = self._clean_strings(columns)
        if not cleaned:
            return "the unresolved target set"
        if len(cleaned) == 1:
            return f"target `{cleaned[0]}`"
        return f"{len(cleaned)} targets ({', '.join(f'`{item}`' for item in cleaned[:6])})"

    def _clean_strings(self, values: Any) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in values or []:
            text = str(raw or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    def _backticked_list(self, values: list[str]) -> str:
        cleaned = self._clean_strings(values)
        if not cleaned:
            return "(not available)"
        preview = cleaned[:8]
        suffix = " ..." if len(cleaned) > 8 else ""
        return ", ".join(f"`{item}`" for item in preview) + suffix

    def _inline_pairs(
        self,
        mapping: dict[str, Any],
        *,
        limit: int,
        skip_keys: set[str] | None = None,
    ) -> str:
        items = self._flatten_mapping(mapping, max_depth=1, skip_keys=skip_keys)
        if not items:
            return "(not available)"
        preview = items[:limit]
        suffix = " ..." if len(items) > limit else ""
        return "; ".join(f"{key}={value}" for key, value in preview) + suffix

    def _preview_text_list(self, values: list[str], *, limit: int) -> str:
        cleaned = [self._english_text(item, default="") for item in values]
        cleaned = [item for item in cleaned if item]
        if not cleaned:
            return "(not available)"
        preview = cleaned[:limit]
        suffix = " ..." if len(cleaned) > limit else ""
        return "; ".join(preview) + suffix

    def _english_list(self, values: list[Any]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            text = self._english_text(value, default="")
            if text:
                cleaned.append(text)
        return cleaned

    def _scalar_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return self._format_number(value)
        return self._english_text(value, default="")

    def _format_number(self, value: int | float) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _safe_text(self, value: Any) -> str:
        text = self._english_text(value, default="")
        return text or "(not available)"

    def _safe_backticked(self, value: Any) -> str:
        text = str(value or "").strip()
        return f"`{text}`" if text else "(not available)"

    def _english_text(self, value: Any, *, default: str) -> str:
        text = self._sanitize_text(value)
        if not text:
            return default
        if self._contains_cjk(text):
            return default
        return text

    def _sanitize_text(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return self._truncate(text, limit=180)

    def _contains_cjk(self, text: str) -> bool:
        return bool(re.search(r"[\u3400-\u9fff]", text))

    def _truncate(self, text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw in values:
            text = self._english_text(raw, default="")
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


async def run_technical_summary(state: GraphState) -> dict:
    """Execute the Technical Summarizer phase and return partial state updates."""

    return await TechnicalSummarizerAgent().run(state)


__all__ = ["TechnicalSummarizerAgent", "run_technical_summary"]
