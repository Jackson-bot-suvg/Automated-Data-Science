"""Reviewer agent that combines hard checks and LLM review comments."""

from __future__ import annotations

from collections.abc import Callable

from app.llm.helpers import render_text
from app.prompts.reviewer_prompt import (
    REVIEWER_COMMENT_SYSTEM_PROMPT,
    build_reviewer_comment_prompt,
)
from app.schemas.reviewer_schema import ReviewCheckResult, ReviewRecord, ReviewStatus
from app.schemas.state_schema import GraphState
from app.tools.phase_trace import (
    phase_record_path,
    register_phase_artifact,
    write_phase_record,
)
from app.tools.run_manifest import write_run_manifest
from app.tools import reviewer_tools


PhaseReviewer = Callable[[GraphState], ReviewCheckResult]


class ReviewerAgent:
    """Review the current phase and return a LangGraph state update."""

    max_retries_per_phase = 3

    phase_reviewers: dict[str, PhaseReviewer] = {
        "preliminary_eda": reviewer_tools.review_preliminary_eda,
        "data_cleaning": reviewer_tools.review_data_cleaning,
        "indepth_eda": reviewer_tools.review_indepth_eda,
        "feature_engineering": reviewer_tools.review_feature_engineering,
        "model_building": reviewer_tools.review_model_building,
    }

    async def run(self, state: GraphState) -> dict:
        """Run deterministic review tools, generate a comment, and update review state."""

        phase = (
            state.current_phase or state.review_state.pending_review_phase or "unknown"
        )
        check_result = self._run_phase_review(phase, state)
        previous_retries = state.review_state.retry_count_by_phase.get(phase, 0)
        attempt = previous_retries + 1
        status = self._status_for(check_result, attempt)

        retry_count_by_phase = dict(state.review_state.retry_count_by_phase)
        if not check_result.passed:
            retry_count_by_phase[phase] = attempt

        fallback_comment = self._fallback_comment(phase, status, check_result)
        comment = await render_text(
            agent_name="reviewer",
            system_prompt=REVIEWER_COMMENT_SYSTEM_PROMPT,
            user_prompt=build_reviewer_comment_prompt(
                phase=phase,
                status=status,
                check_result=check_result,
            ),
            fallback=fallback_comment,
        )

        history_record = ReviewRecord(
            phase=phase,
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
                f"[{phase}] {failure}" for failure in check_result.blocking_failures
            )

        review_record_path = phase_record_path(
            workspace_root=state.runtime_state.workspace_root,
            artifact_root=state.runtime_state.artifact_root,
            category="phase_reviews",
            phase=phase,
        )
        write_phase_record(
            review_record_path,
            {
                "phase": phase,
                "status": status,
                "attempt": attempt,
                "blocking_failures": list(check_result.blocking_failures),
                "warnings": list(check_result.warnings),
                "checked_artifacts": list(check_result.checked_artifacts),
                "checked_state_fields": list(check_result.checked_state_fields),
                "comment": comment,
                "details": check_result.details,
            },
        )
        artifacts = register_phase_artifact(
            state.artifacts,
            bucket="phase_review_records",
            phase=phase,
            path=review_record_path,
        )
        manifest_path = str(state.artifacts.get("run_manifest") or "")
        if manifest_path:
            write_run_manifest(
                workspace_root=state.runtime_state.workspace_root,
                manifest_path=manifest_path,
                run_id=state.runtime_state.run_id or state.task_config.task_id,
                thread_id=state.runtime_state.thread_id,
                current_phase=phase,
                phase_history=list(state.phase_history or []) + [phase],
                task_config=state.task_config.model_dump(mode="python"),
                target_spec=state.target_spec.model_dump(mode="python"),
                artifacts=artifacts,
                runtime_state=state.runtime_state.model_dump(mode="python"),
                data_state=state.data_state.model_dump(mode="python"),
                feature_state=state.feature_state.model_dump(mode="python"),
                model_state=state.model_state.model_dump(mode="python"),
                review_state={
                    **state.review_state.model_dump(mode="python"),
                    "latest_review_status": status,
                    "last_review_comment": comment,
                },
                report_state=state.report_state.model_dump(mode="python"),
            )

        return {
            "review_state": {
                **state.review_state.model_dump(),
                "latest_review_status": status,
                "review_history": list(state.review_state.review_history)
                + [history_record],
                "retry_count_by_phase": retry_count_by_phase,
                "fatal_errors": fatal_errors,
                "last_review_comment": comment,
                "pending_review_phase": phase,
            },
            "runtime_state": {
                **state.runtime_state.model_dump(),
                "interrupted": status == "fatal",
                "last_reviewed_phase": phase,
            },
            "artifacts": artifacts,
        }

    def _run_phase_review(self, phase: str, state: GraphState) -> ReviewCheckResult:
        review_func = self.phase_reviewers.get(phase)
        if review_func is None:
            return ReviewCheckResult(
                phase=phase,
                blocking_failures=[f"reviewer does not support phase '{phase}'"],
                checked_state_fields=["current_phase"],
            )
        return review_func(state)

    def _status_for(
        self, check_result: ReviewCheckResult, attempt: int
    ) -> ReviewStatus:
        if check_result.passed:
            return "approved"
        if attempt >= self.max_retries_per_phase:
            return "fatal"
        return "retry"

    def _fallback_comment(
        self,
        phase: str,
        status: ReviewStatus,
        check_result: ReviewCheckResult,
    ) -> str:
        if status == "approved":
            warning_text = "; ".join(check_result.warnings) or "no blocking issues"
            return f"{phase} approved: {warning_text}"
        failure_text = "; ".join(check_result.blocking_failures)
        return f"{phase} {status}: {failure_text}"


__all__ = ["ReviewerAgent"]
