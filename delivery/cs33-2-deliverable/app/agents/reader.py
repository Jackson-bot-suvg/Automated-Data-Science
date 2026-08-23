"""Main-chain Reader agent for initial task and dataset understanding."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.llm.helpers import complete_json
from app.prompts.reader_prompt import READER_SYSTEM_PROMPT, build_reader_prompt
from app.schemas.state_schema import (
    EvaluationSpec,
    GraphState,
    SubmissionSpec,
    TargetEvidence,
    TargetSpec,
)
from app.tools.profiler import profile_dataset
from app.utils.target_scope import (
    canonical_target_scope,
    exact_target_matches,
    normalize_target_columns,
)
from app.utils.task_contract import (
    canonical_candidate_metrics,
    infer_task_family_from_targets,
    normalize_target_mode,
    normalize_task_family,
)


class ReaderAgent:
    """Convert incoming task config and dataset paths into workflow state."""

    preview_rows = 1000

    async def run(self, state: GraphState) -> dict:
        """Run Reader and return a serializable LangGraph state update."""

        workspace_root = self._workspace_root(state)
        discovered_paths = self._discover_dataset_paths(state, workspace_root)
        train_path = self._resolve_path(workspace_root, discovered_paths["train"])
        test_path = self._resolve_path(workspace_root, discovered_paths["test"])
        submission_path = self._resolve_path(
            workspace_root, discovered_paths["submission"]
        )
        overview_text = self._bounded_overview_text(discovered_paths["overview_files"])

        train_meta = self._csv_metadata(train_path)
        test_meta = self._csv_metadata(test_path)
        submission_meta = self._csv_metadata(submission_path)

        resolved_target = self._resolve_target_scope(
            state,
            train_meta,
            test_meta,
            submission_meta,
            overview_text,
        )
        _, target_columns = canonical_target_scope(
            target_columns=resolved_target["target_columns"]
        )
        task_type = self._task_type(state, train_meta, target_columns)
        candidate_metrics = self._candidate_metrics(state, task_type, overview_text)
        target_spec = self._build_target_spec(
            state=state,
            resolved_target=resolved_target,
            task_type=task_type,
            candidate_metrics=candidate_metrics,
            train_meta=train_meta,
            submission_meta=submission_meta,
        )
        target_column, target_columns = canonical_target_scope(
            target_columns=target_spec["target_columns"]
        )
        risks = self._risks(
            state,
            train_meta,
            test_meta,
            submission_meta,
            target_column,
            target_columns,
        )
        data_overview = self._data_overview(train_meta)

        updated_task_config = state.task_config.model_dump(mode="python")
        updated_task_config["dataset_paths"] = {
            "train": train_meta["path"],
            "test": test_meta["path"],
            "submission": submission_meta["path"],
        }
        updated_task_config["target_column"] = ""
        updated_task_config["target_columns"] = []

        fallback = {
            "task_summary": self._task_summary(
                state, task_type, target_column, target_columns
            ),
            "predicted_task_type": task_type,
            "target_column": target_column,
            "target_columns": list(target_columns),
            "candidate_metrics": candidate_metrics,
            "risks": risks,
            "business_goal_summary": self._business_goal_summary(state),
        }
        reader_context = {
            "task_config": state.task_config.model_dump(mode="python"),
            "workspace_root": str(workspace_root),
            "dataset_files": discovered_paths["files"],
            "overview_text": overview_text,
            "train_metadata": self._prompt_metadata(train_meta),
            "test_metadata": self._prompt_metadata(test_meta),
            "submission_metadata": self._prompt_metadata(submission_meta),
            "target_spec": target_spec,
        }
        llm_payload = await complete_json(
            agent_name="reader",
            system_prompt=READER_SYSTEM_PROMPT,
            user_prompt=build_reader_prompt(
                reader_context=reader_context, fallback=fallback
            ),
            fallback=fallback,
        )
        problem_understanding = self._problem_understanding(
            llm_payload,
            fallback,
            data_overview,
            list(train_meta.get("columns") or []),
            target_spec,
        )
        target_spec["task_family"] = normalize_task_family(
            problem_understanding["predicted_task_type"],
            target_mode=target_spec.get("target_mode", ""),
            target_count=len(target_spec.get("target_columns") or []),
        )
        target_spec["evaluation"]["candidate_metrics"] = canonical_candidate_metrics(
            target_spec["task_family"],
            problem_understanding["candidate_metrics"],
        )
        target_spec["evaluation"]["primary_metric"] = (
            target_spec["evaluation"]["candidate_metrics"][0]
            if target_spec["evaluation"]["candidate_metrics"]
            else target_spec["evaluation"]["primary_metric"]
        )
        resolved_target_column, resolved_target_columns = canonical_target_scope(
            target_columns=target_spec["target_columns"]
        )

        eda_findings = {}
        if train_meta.get("readable") and train_path:
            try:
                df_train = pd.read_csv(train_path)
                eda_findings = profile_dataset(
                    df_train,
                    target_cols=resolved_target_columns,
                )
            except Exception as exc:
                eda_findings = {"error": str(exc)}

        return {
            "task_config": updated_task_config,
            "problem_understanding": problem_understanding,
            "target_spec": target_spec,
            "data_state": {
                **state.data_state.model_dump(mode="python"),
                "raw_train_path": train_meta["path"],
                "raw_test_path": test_meta["path"],
                "current_train_path": train_meta["path"],
                "current_test_path": test_meta["path"],
                "target_column": "",
                "target_columns": [],
                "train_shape": train_meta["shape"],
                "test_shape": test_meta["shape"],
                "eda_findings": eda_findings,
                "schema_summary": {
                    **state.data_state.schema_summary,
                    "dataset_files": discovered_paths["files"],
                    "overview_files": [
                        str(path) for path in discovered_paths["overview_files"]
                    ],
                    "sample_submission_path": submission_meta["path"],
                },
            },
            "current_phase": "reader",
            "phase_history": ["reader"],
            "messages": [
                {
                    "role": "system",
                    "name": "reader",
                    "content": "Initialized problem understanding and dataset paths.",
                }
            ],
        }

    def _workspace_root(self, state: GraphState) -> Path:
        return Path(state.runtime_state.workspace_root or ".").resolve()

    def _resolve_path(self, workspace_root: Path, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = workspace_root / path
        return path.resolve()

    def _discover_dataset_paths(
        self, state: GraphState, workspace_root: Path
    ) -> dict[str, Any]:
        dataset_paths = state.task_config.dataset_paths
        discovered = {
            "train": dataset_paths.train,
            "test": dataset_paths.test,
            "submission": dataset_paths.submission,
            "overview_files": [],
            "files": [],
        }

        root_raw = state.task_config.dataset_root
        root = self._resolve_path(workspace_root, root_raw) if root_raw else None
        if root and root.is_file() and not discovered["train"]:
            discovered["train"] = str(root)
            return discovered
        if not root or not root.is_dir():
            return discovered

        children = [path for path in root.iterdir() if path.is_file()]
        discovered["files"] = sorted(path.name for path in children)
        csv_by_name = {
            path.name.lower(): path
            for path in children
            if path.suffix.lower() == ".csv"
        }

        if not discovered["train"]:
            discovered["train"] = str(
                self._first_csv(csv_by_name, ("train.csv",), "train") or ""
            )
        if not discovered["test"]:
            discovered["test"] = str(
                self._first_csv(csv_by_name, ("test.csv",), "test") or ""
            )
        if not discovered["submission"]:
            discovered["submission"] = str(
                self._first_csv(
                    csv_by_name,
                    (
                        "sample_submission.csv",
                        "submission.csv",
                        "gender_submission.csv",
                    ),
                    "submission",
                )
                or ""
            )
        discovered["overview_files"] = [
            path
            for path in children
            if path.name.lower() in {"overview.txt", "data_description.txt"}
        ]
        return discovered

    def _first_csv(
        self,
        csv_by_name: dict[str, Path],
        exact_names: tuple[str, ...],
        contains: str,
    ) -> Path | None:
        for name in exact_names:
            if name in csv_by_name:
                return csv_by_name[name]
        matches = sorted(
            path for name, path in csv_by_name.items() if contains in name
        )
        return matches[0] if matches else None

    def _bounded_overview_text(self, paths: list[Path], *, max_chars: int = 6000) -> str:
        parts: list[str] = []
        remaining = max_chars
        for path in paths:
            if remaining <= 0:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            excerpt = text[:remaining]
            parts.append(f"{path.name}:\n{excerpt}")
            remaining -= len(excerpt)
        return "\n\n".join(parts)

    def _csv_metadata(self, path: Path | None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "path": str(path) if path else "",
            "exists": bool(path and path.exists()),
            "readable": False,
            "shape": None,
            "columns": [],
            "column_types": {},
            "missingness_summary": {},
            "target_sample": {},
            "error": "",
        }
        if not path or not path.exists():
            metadata["error"] = "path is empty or does not exist"
            return metadata

        try:
            sample = pd.read_csv(path, nrows=self.preview_rows)
        except Exception as exc:
            metadata["error"] = str(exc)
            return metadata

        metadata["readable"] = True
        metadata["columns"] = [str(column) for column in sample.columns]
        metadata["column_types"] = {
            str(column): str(dtype) for column, dtype in sample.dtypes.items()
        }
        metadata["missingness_summary"] = {
            str(column): int(count)
            for column, count in sample.isna().sum().items()
            if int(count) > 0
        }
        metadata["shape"] = self._csv_shape(path, sample)
        metadata["sample_n_rows"] = int(sample.shape[0])
        metadata["sample_n_cols"] = int(sample.shape[1])
        metadata["sample"] = sample.head(5).to_dict(orient="records")
        return metadata

    def _csv_shape(self, path: Path, sample: pd.DataFrame) -> tuple[int, int]:
        if sample.empty:
            return (0, int(sample.shape[1]))
        first_column = sample.columns[0]
        try:
            row_count = sum(
                len(chunk)
                for chunk in pd.read_csv(path, usecols=[first_column], chunksize=50000)
            )
        except Exception:
            row_count = int(sample.shape[0])
        return (int(row_count), int(sample.shape[1]))

    def _resolve_target_scope(
        self,
        state: GraphState,
        train_meta: dict[str, Any],
        test_meta: dict[str, Any],
        submission_meta: dict[str, Any],
        overview_text: str,
    ) -> dict[str, Any]:
        explicit_targets = self._explicit_target_columns(state)
        if explicit_targets:
            return self._resolved_target(
                explicit_targets,
                source="compatibility_fields",
                detail="Used explicit target columns provided in the incoming state.",
            )

        train_columns = normalize_target_columns(train_meta.get("columns") or [])
        structured_targets, structured_evidence = self._structured_target_columns(
            train_meta, test_meta, submission_meta
        )
        if structured_targets:
            return {
                "status": "resolved",
                "target_columns": structured_targets,
                "evidence": structured_evidence,
            }

        overview_targets = self._targets_from_overview(overview_text, train_columns)
        if overview_targets:
            return self._resolved_target(
                overview_targets,
                source="overview",
                detail="Resolved target columns from dataset overview or data description.",
            )

        hint_text = " ".join(
            part
            for part in (
                state.task_config.target_hint,
                state.task_config.raw_task,
                state.task_config.business_context,
            )
            if part
        )
        hinted_targets = self._target_columns_from_hint(hint_text, train_columns)
        if hinted_targets:
            return self._resolved_target(
                hinted_targets,
                source="user_hint",
                detail="Resolved target columns from user-provided natural language hints.",
            )
        if "Survived" in train_columns:
            return self._resolved_target(
                ["Survived"],
                source="train_schema",
                detail="Used the common Titanic target column fallback from training schema.",
            )
        return {
            "status": "unresolved",
            "target_columns": [],
            "evidence": [
                self._evidence(
                    source="unresolved",
                    detail=(
                        "Could not resolve target columns from compatibility fields, "
                        "train/test differences, sample submission, overview, or user hints."
                    ),
                )
            ],
        }

    def _explicit_target_columns(self, state: GraphState) -> list[str]:
        explicit_columns = (
            state.target_spec.target_columns
            or state.task_config.target_columns
        )
        explicit_column = (
            canonical_target_scope(target_columns=state.target_spec.target_columns)[0]
            or state.task_config.target_column
        )
        _, resolved_columns = canonical_target_scope(explicit_column, explicit_columns)
        return resolved_columns

    def _structured_target_columns(
        self,
        train_meta: dict[str, Any],
        test_meta: dict[str, Any],
        submission_meta: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        train_columns = normalize_target_columns(train_meta.get("columns") or [])
        if not train_columns:
            return [], []
        test_columns = set(test_meta.get("columns") or [])
        submission_columns = [
            column
            for column in normalize_target_columns(submission_meta.get("columns") or [])
            if column.lower() not in {"id", "index", "passengerid"}
        ]
        train_test_diff = [column for column in train_columns if column not in test_columns]
        submission_intersection = [
            column for column in submission_columns if column in train_test_diff
        ]
        evidence: list[dict[str, Any]] = []
        if train_test_diff:
            evidence.append(
                self._evidence(
                    source="train_test_diff",
                    detail=(
                        "Columns present in train.csv but missing from test.csv suggest "
                        "candidate targets."
                    ),
                    columns=train_test_diff,
                )
            )
        if submission_intersection:
            evidence.append(
                self._evidence(
                    source="sample_submission",
                    detail="Sample submission output columns align with train/test target diff.",
                    columns=submission_intersection,
                )
            )
            return submission_intersection, evidence
        if len(train_test_diff) == 1:
            return train_test_diff, evidence
        if len(submission_columns) == 1 and submission_columns[0] in train_columns:
            evidence.append(
                self._evidence(
                    source="sample_submission",
                    detail="Single sample submission output column exists in training data.",
                    columns=submission_columns,
                )
            )
            return submission_columns, evidence
        return [], evidence

    def _target_columns_from_hint(
        self, hint: str, train_columns: list[str]
    ) -> list[str]:
        if not hint or not train_columns:
            return []
        exact_matches = exact_target_matches(hint, train_columns)
        if exact_matches:
            return exact_matches

        hint_text = hint.lower()
        normalized_hint = self._normalize_text(hint_text)
        scored: list[tuple[int, str]] = []
        for column in train_columns:
            words = [word for word in re.split(r"[_\W]+", column.lower()) if word]
            score = sum(1 for word in words if word in normalized_hint)
            if score:
                scored.append((score, column))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            return [scored[0][1]]
        return []

    def _targets_from_overview(
        self, overview_text: str, train_columns: list[str]
    ) -> list[str]:
        if not overview_text:
            return []
        text = overview_text.lower()
        exact_matches = exact_target_matches(text, train_columns)
        if len(exact_matches) > 1 and any(
            marker in text
            for marker in (
                "target columns",
                "target labels",
                "label columns",
                "multi-label",
                "multi target",
            )
        ):
            return exact_matches
        for column in train_columns:
            col = re.escape(column.lower())
            patterns = (
                rf"target\s+(?:variable|column)?\s*`?{col}`?",
                rf"`?{col}`?\s+is\s+the\s+(?:binary\s+)?target",
                rf"predict\s+(?:the\s+)?(?:probability\s+of\s+)?`?{col}`?",
            )
            if any(re.search(pattern, text) for pattern in patterns):
                return [column]
        return []

    def _normalize_text(self, value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    def _task_type(
        self, state: GraphState, train_meta: dict[str, Any], target_columns: list[str]
    ) -> str:
        explicit = state.task_config.task_type
        if explicit:
            return normalize_task_family(
                explicit,
                target_mode=normalize_target_mode(
                    state.target_spec.target_mode,
                    target_count=len(target_columns),
                    task_family=explicit,
                ),
                target_count=len(target_columns),
            )

        train_path = str(train_meta.get("path") or "")
        if train_path and target_columns:
            try:
                target_frame = pd.read_csv(train_path, usecols=target_columns, nrows=1000)
                inferred = infer_task_family_from_targets(target_frame, target_columns)
                if inferred != "unknown":
                    return inferred
            except Exception:
                pass

        if len(target_columns) > 1:
            return "multilabel_classification"
        return "classification_unspecified"

    def _target_values_for_inference(
        self, train_meta: dict[str, Any], target_column: str, *, max_rows: int = 1000
    ) -> pd.Series:
        sample = pd.DataFrame(train_meta.get("sample") or [])
        if target_column in sample.columns:
            values = sample[target_column].dropna()
            if not values.empty:
                return values

        train_path = str(train_meta.get("path") or "")
        if not train_path:
            return pd.Series(dtype="object")
        try:
            sampled_column = pd.read_csv(
                train_path, usecols=[target_column], nrows=max_rows
            )
        except Exception:
            return pd.Series(dtype="object")
        return sampled_column[target_column].dropna()

    def _looks_regression_like_numeric(self, values: pd.Series) -> bool:
        if values.empty or not pd.api.types.is_numeric_dtype(values):
            return False

        unique_count = int(values.nunique())
        sample_size = int(values.shape[0])
        if sample_size <= 0:
            return False

        # Fractional numeric targets are almost always regression.
        has_fractional_values = ((values % 1) != 0).any()
        if has_fractional_values:
            return True

        unique_ratio = unique_count / sample_size
        if unique_count >= 20:
            return True
        if unique_count >= 5 and unique_ratio >= 0.5:
            return True
        return False

    def _candidate_metrics(
        self, state: GraphState, task_type: str, overview_text: str = ""
    ) -> list[str]:
        if state.task_config.metric_hint:
            return canonical_candidate_metrics(
                task_type,
                [
                    metric.strip()
                    for metric in state.task_config.metric_hint.split(",")
                    if metric.strip()
                ],
            )
        if "roc" in overview_text.lower() or "auc" in overview_text.lower():
            hinted = canonical_candidate_metrics(task_type, ["roc_auc", "accuracy", "f1"])
            return hinted
        return canonical_candidate_metrics(task_type, [])

    def _build_target_spec(
        self,
        *,
        state: GraphState,
        resolved_target: dict[str, Any],
        task_type: str,
        candidate_metrics: list[str],
        train_meta: dict[str, Any],
        submission_meta: dict[str, Any],
    ) -> dict[str, Any]:
        target_columns = normalize_target_columns(resolved_target.get("target_columns") or [])
        target_mode = normalize_target_mode(
            state.target_spec.target_mode,
            target_count=len(target_columns),
            task_family=task_type,
        )
        canonical_task_family = normalize_task_family(
            task_type,
            target_mode=target_mode,
            target_count=len(target_columns),
        )
        candidate_metrics = canonical_candidate_metrics(canonical_task_family, candidate_metrics)

        target_spec = TargetSpec(
            status="resolved" if target_columns else "unresolved",
            target_columns=target_columns,
            target_mode=target_mode,
            task_family=canonical_task_family,
            prediction_kind=self._prediction_kind(
                canonical_task_family, target_columns, train_meta, submission_meta
            ),
            user_target_hint=state.task_config.target_hint,
            user_task_family_hint=state.task_config.task_type,
            submission=SubmissionSpec(
                **self._submission_spec(target_columns, submission_meta)
            ),
            evaluation=EvaluationSpec(
                metric_hint=state.task_config.metric_hint,
                primary_metric=candidate_metrics[0] if candidate_metrics else "",
                candidate_metrics=list(candidate_metrics),
            ),
            evidence=[
                TargetEvidence(**item)
                for item in (resolved_target.get("evidence") or [])
            ],
        )
        return target_spec.model_dump(mode="python")

    def _submission_spec(
        self, target_columns: list[str], submission_meta: dict[str, Any]
    ) -> dict[str, Any]:
        submission_columns = normalize_target_columns(submission_meta.get("columns") or [])
        target_set = set(target_columns)
        output_columns = [
            column for column in submission_columns if column in target_set
        ]
        id_columns = [
            column for column in submission_columns if column not in target_set
        ]
        return SubmissionSpec(
            id_columns=id_columns,
            output_columns=output_columns,
            submission_columns=submission_columns,
            sample_submission_path=submission_meta.get("path") or "",
        ).model_dump(mode="python")

    def _prediction_kind(
        self,
        task_type: str,
        target_columns: list[str],
        train_meta: dict[str, Any],
        submission_meta: dict[str, Any],
    ) -> str:
        if not target_columns:
            return ""
        if task_type == "regression":
            return "numeric"

        submission_sample = pd.DataFrame(submission_meta.get("sample") or [])
        for column in target_columns:
            if column not in submission_sample.columns:
                continue
            values = submission_sample[column].dropna()
            if values.empty:
                continue
            if pd.api.types.is_numeric_dtype(values):
                fractional = (values % 1 != 0).any()
                bounded = values.between(0, 1).all()
                if fractional or bounded:
                    return "probability"

        train_sample = pd.DataFrame(train_meta.get("sample") or [])
        for column in target_columns:
            values = self._target_values_for_inference(train_meta, column)
            if self._looks_regression_like_numeric(values):
                return "numeric"
        return "label"

    def _resolved_target(
        self, columns: list[str], *, source: str, detail: str
    ) -> dict[str, Any]:
        return {
            "status": "resolved",
            "target_columns": normalize_target_columns(columns),
            "evidence": [self._evidence(source=source, detail=detail, columns=columns)],
        }

    def _evidence(
        self, *, source: str, detail: str, columns: list[str] | None = None
    ) -> dict[str, Any]:
        return TargetEvidence(
            source=source,
            detail=detail,
            columns=normalize_target_columns(columns),
        ).model_dump(mode="python")

    def _risks(
        self,
        state: GraphState,
        train_meta: dict[str, Any],
        test_meta: dict[str, Any],
        submission_meta: dict[str, Any],
        target_column: str,
        target_columns: list[str],
    ) -> list[str]:
        risks: list[str] = []
        resolved_target_column, resolved_target_columns = canonical_target_scope(
            target_column, target_columns
        )
        if not train_meta.get("readable"):
            risks.append(f"Training dataset is not readable: {train_meta.get('error')}")
        if test_meta.get("path") and not test_meta.get("readable"):
            risks.append(f"Test dataset is not readable: {test_meta.get('error')}")
        if not test_meta.get("path"):
            risks.append(
                "No external test.csv was found; evaluation must rely on train-only validation"
            )
        if not submission_meta.get("path"):
            risks.append("No sample submission file was found")
        if not resolved_target_columns:
            risks.append(
                "Target scope is not specified and could not be inferred from structured dataset evidence"
            )
        elif train_meta.get("readable"):
            train_columns = set(train_meta.get("columns") or [])
            missing_targets = [
                column for column in resolved_target_columns if column not in train_columns
            ]
            if len(missing_targets) == 1:
                risks.append(
                    f"Target column '{missing_targets[0]}' is missing from training dataset"
                )
            elif missing_targets:
                risks.append(
                    "Target columns are missing from training dataset: "
                    + ", ".join(missing_targets)
                )
        if train_meta.get("missingness_summary"):
            risks.append("Training dataset contains missing values in sampled rows")
        return risks

    def _data_overview(self, train_meta: dict[str, Any]) -> dict[str, Any]:
        shape = train_meta.get("shape") or (0, 0)
        return {
            "n_rows": int(shape[0]),
            "n_cols": int(shape[1]),
            "column_types": dict(train_meta.get("column_types") or {}),
            "missingness_summary": dict(train_meta.get("missingness_summary") or {}),
        }

    def _task_summary(
        self,
        state: GraphState,
        task_type: str,
        target_column: str,
        target_columns: list[str],
    ) -> str:
        _, resolved_target_columns = canonical_target_scope(target_column, target_columns)
        if not resolved_target_columns:
            target_text = "an unspecified target"
        elif len(resolved_target_columns) == 1:
            target_text = resolved_target_columns[0]
        else:
            preview = ", ".join(resolved_target_columns[:3])
            suffix = "..." if len(resolved_target_columns) > 3 else ""
            target_text = (
                f"{len(resolved_target_columns)} target columns "
                f"({preview}{suffix})"
            )
        if state.task_config.task_id:
            return f"{state.task_config.task_id}: {task_type} task predicting {target_text}."
        return f"{task_type} task predicting {target_text}."

    def _business_goal_summary(self, state: GraphState) -> str:
        if state.task_config.business_context:
            return state.task_config.business_context
        return "Produce a validated data science workflow and final report."

    def _problem_understanding(
        self,
        payload: dict[str, Any],
        fallback: dict[str, Any],
        data_overview: dict[str, Any],
        train_columns: list[str],
        target_spec: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_columns = set(train_columns)
        payload_target_columns = self._list_of_strings(payload.get("target_columns"), [])
        if allowed_columns:
            payload_target_columns = [
                column for column in payload_target_columns if column in allowed_columns
            ]
        payload_target_column = self._string(payload.get("target_column"), "")
        if allowed_columns and payload_target_column not in allowed_columns:
            payload_target_column = ""
        resolved_target_column, resolved_target_columns = canonical_target_scope(
            payload_target_column,
            payload_target_columns,
        )
        if not resolved_target_columns:
            resolved_target_column, resolved_target_columns = canonical_target_scope(
                target_columns=target_spec["target_columns"]
            )
        return {
            "task_summary": self._string(
                payload.get("task_summary"), fallback["task_summary"]
            ),
            "predicted_task_type": normalize_task_family(
                self._string(
                    payload.get("predicted_task_type"), fallback["predicted_task_type"]
                ),
                target_mode=target_spec.get("target_mode", ""),
                target_count=len(target_spec.get("target_columns") or []),
            ),
            "target_column": "",
            "target_columns": [],
            "candidate_metrics": canonical_candidate_metrics(
                target_spec.get("task_family", ""),
                self._list_of_strings(
                    payload.get("candidate_metrics"), fallback["candidate_metrics"]
                ),
            ),
            "data_overview": data_overview,
            "risks": self._list_of_strings(payload.get("risks"), fallback["risks"]),
            "business_goal_summary": self._string(
                payload.get("business_goal_summary"),
                fallback["business_goal_summary"],
            ),
        }

    def _prompt_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": metadata.get("path", ""),
            "exists": metadata.get("exists", False),
            "readable": metadata.get("readable", False),
            "shape": metadata.get("shape"),
            "columns": metadata.get("columns", []),
            "column_types": metadata.get("column_types", {}),
            "missingness_summary": metadata.get("missingness_summary", {}),
            "sample": metadata.get("sample", []),
            "error": metadata.get("error", ""),
        }

    def _string(self, value: Any, default: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default

    def _list_of_strings(self, value: Any, default: list[str]) -> list[str]:
        if not isinstance(value, list):
            return list(default)
        cleaned = [str(item) for item in value if str(item).strip()]
        return cleaned or list(default)


async def run_reader(state: GraphState) -> dict:
    """Execute the Reader phase and return partial state updates."""

    return await ReaderAgent().run(state)


__all__ = ["ReaderAgent", "run_reader"]
