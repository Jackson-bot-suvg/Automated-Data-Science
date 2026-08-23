"""Background reviewer agent for validating Reader output before planning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.llm.helpers import complete_json
from app.prompts.reviewer_prompt import (
    BACKGROUND_REVIEWER_SYSTEM_PROMPT,
    build_background_reviewer_comment_prompt,
)
from app.schemas.reviewer_schema import ReviewCheckResult, ReviewRecord, ReviewStatus
from app.schemas.state_schema import GraphState


class BackgroundReviewerAgent:
    """Review Reader output and decide whether the graph can enter planning."""

    max_retries_per_phase = 3
    reviewed_phase = "reader"

    async def run(self, state: GraphState) -> dict:
        """Run deterministic checks, optionally phrase a comment, and update review state."""

        check_result, hard_fatal = self._check_reader_output(state)
        previous_retries = state.review_state.retry_count_by_phase.get(
            self.reviewed_phase, 0
        )
        attempt = previous_retries + 1
        status = self._status_for(check_result, attempt, hard_fatal)
        fallback_comment = self._fallback_comment(status, check_result)
        comment = await self._llm_comment(status, check_result, fallback_comment)

        retry_count_by_phase = dict(state.review_state.retry_count_by_phase)
        if status == "retry":
            retry_count_by_phase[self.reviewed_phase] = attempt
        elif status == "fatal" and not check_result.passed:
            retry_count_by_phase[self.reviewed_phase] = attempt

        history_record = ReviewRecord(
            phase=self.reviewed_phase,
            status=status,
            attempt=attempt,
            blocking_failures=check_result.blocking_failures,
            warnings=check_result.warnings,
            checked_artifacts=check_result.checked_artifacts,
            checked_state_fields=check_result.checked_state_fields,
            comment=comment,
            details=check_result.details,
        ).model_dump(mode="python")

        fatal_errors = list(state.review_state.fatal_errors)
        if status == "fatal":
            fatal_errors.extend(
                f"[reader] {failure}" for failure in check_result.blocking_failures
            )

        return {
            "review_state": {
                **state.review_state.model_dump(mode="python"),
                "latest_review_status": status,
                "review_history": list(state.review_state.review_history)
                + [history_record],
                "retry_count_by_phase": retry_count_by_phase,
                "fatal_errors": fatal_errors,
                "last_review_comment": comment,
                "pending_review_phase": self.reviewed_phase,
            },
            "runtime_state": {
                **state.runtime_state.model_dump(mode="python"),
                "interrupted": status == "fatal",
                "last_reviewed_phase": self.reviewed_phase,
            },
        }

    def _check_reader_output(self, state: GraphState) -> tuple[ReviewCheckResult, bool]:
        result = ReviewCheckResult(phase=self.reviewed_phase)
        hard_fatal = False
        workspace_root = Path(state.runtime_state.workspace_root or ".").resolve()

        train_path = (
            state.data_state.current_train_path or state.task_config.dataset_paths.train
        )
        test_path = state.data_state.current_test_path or state.task_config.dataset_paths.test
        submission_path = (
            state.target_spec.submission.sample_submission_path
            or state.task_config.dataset_paths.submission
        )

        result.checked_state_fields.extend(
            [
                "problem_understanding.task_summary",
                "problem_understanding.candidate_metrics",
                "problem_understanding.data_overview",
                "target_spec.status",
                "target_spec.target_columns",
                "target_spec.target_mode",
                "target_spec.submission.submission_columns",
                "data_state.current_train_path",
                "data_state.current_test_path",
            ]
        )

        train_resolved = self._resolve_path(workspace_root, train_path)
        train_columns = self._csv_columns(train_resolved)
        if not train_resolved or not train_resolved.exists():
            result.blocking_failures.append(
                f"training dataset is missing at {train_path or '(empty)'}"
            )
            hard_fatal = True
        elif not train_columns:
            result.blocking_failures.append(
                f"training dataset is unreadable at {train_resolved}"
            )
            hard_fatal = True
        else:
            result.details["train_columns"] = train_columns

        test_columns: list[str] = []
        if test_path:
            test_resolved = self._resolve_path(workspace_root, test_path)
            test_columns = self._csv_columns(test_resolved)
            if not test_resolved or not test_resolved.exists():
                result.warnings.append(f"test dataset is missing at {test_path}")
            elif not test_columns:
                result.warnings.append(f"test dataset is unreadable at {test_resolved}")
            else:
                result.details["test_columns"] = test_columns
        else:
            result.warnings.append(
                "test dataset is not configured; workflow may continue with train-only validation"
            )

        submission_columns: list[str] = []
        if submission_path:
            submission_resolved = self._resolve_path(workspace_root, submission_path)
            submission_columns = self._csv_columns(submission_resolved)
            if not submission_resolved or not submission_resolved.exists():
                result.warnings.append(
                    f"sample submission is missing at {submission_path}"
                )
            elif not submission_columns:
                result.warnings.append(
                    f"sample submission is unreadable at {submission_resolved}"
                )
            else:
                result.details["sample_submission_columns"] = submission_columns
        else:
            result.warnings.append("sample submission is not configured")

        if not state.problem_understanding.task_summary:
            result.blocking_failures.append(
                "problem_understanding.task_summary is empty"
            )

        result.details["target_spec"] = state.target_spec.model_dump(mode="python")
        target_columns = list(state.target_spec.target_columns)
        result.details["target_columns"] = list(target_columns)

        if state.target_spec.status != "resolved" or not target_columns:
            result.blocking_failures.append(
                "target unresolved: Reader must return target_spec.status=resolved with concrete target columns"
            )
        elif train_columns:
            missing_targets = [
                column for column in target_columns if column not in train_columns
            ]
            if len(missing_targets) == 1:
                result.blocking_failures.append(
                    f"target unresolved: training dataset is missing target column '{missing_targets[0]}'"
                )
            elif missing_targets:
                result.blocking_failures.append(
                    "target unresolved: training dataset is missing target columns "
                    + ", ".join(missing_targets)
                )

        if test_columns and target_columns:
            leaked_targets = [
                column for column in target_columns if column in test_columns
            ]
            if leaked_targets:
                result.blocking_failures.append(
                    "target leakage: test dataset contains target columns "
                    + ", ".join(leaked_targets)
                )

        expected_submission = list(state.target_spec.submission.submission_columns)
        if expected_submission and submission_columns and submission_columns != expected_submission:
            result.blocking_failures.append(
                "submission mismatch: sample submission columns do not match expected order "
                + f"{expected_submission}"
            )

        if not state.problem_understanding.candidate_metrics:
            result.warnings.append(
                "candidate_metrics is empty; planner can continue but reporting will be weaker"
            )
        if not state.problem_understanding.data_overview.column_types:
            result.warnings.append("data_overview.column_types is empty")

        return result, hard_fatal

    def _resolve_path(self, workspace_root: Path, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = workspace_root / path
        return path.resolve()

    def _csv_columns(self, path: Path | None) -> list[str]:
        if not path or not path.exists():
            return []
        try:
            return [str(column) for column in pd.read_csv(path, nrows=0).columns]
        except Exception:
            return []

    def _status_for(
        self,
        check_result: ReviewCheckResult,
        attempt: int,
        hard_fatal: bool,
    ) -> ReviewStatus:
        if check_result.passed:
            return "approved"
        if hard_fatal or attempt >= self.max_retries_per_phase:
            return "fatal"
        return "retry"

    def _fallback_comment(
        self, status: ReviewStatus, check_result: ReviewCheckResult
    ) -> str:
        if status == "approved":
            warning_text = (
                "; ".join(check_result.warnings)
                or "Reader output is ready for planning"
            )
            return f"reader approved: {warning_text}"
        failure_text = "; ".join(check_result.blocking_failures)
        return f"reader {status}: {failure_text}"

    async def _llm_comment(
        self,
        status: ReviewStatus,
        check_result: ReviewCheckResult,
        fallback_comment: str,
    ) -> str:
        payload = await complete_json(
            agent_name="bg_reviewer",
            system_prompt=BACKGROUND_REVIEWER_SYSTEM_PROMPT,
            user_prompt=build_background_reviewer_comment_prompt(
                status=status,
                check_result=check_result,
                fallback_comment=fallback_comment,
            ),
            fallback={"comment": fallback_comment},
        )
        comment = payload.get("comment")
        if isinstance(comment, str) and comment.strip():
            return comment.strip()
        return fallback_comment


async def run_bg_reviewer(state: GraphState) -> dict:
    """Execute background review and return partial state updates."""

    return await BackgroundReviewerAgent().run(state)


__all__ = ["BackgroundReviewerAgent", "run_bg_reviewer"]
