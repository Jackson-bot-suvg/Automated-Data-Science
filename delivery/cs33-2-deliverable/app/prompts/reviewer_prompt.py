"""Prompt assembly for LLM reviewer comments."""

from __future__ import annotations

from app.schemas.reviewer_schema import ReviewCheckResult, ReviewStatus


REVIEWER_COMMENT_SYSTEM_PROMPT = """
You are a data science workflow reviewer.
The deterministic checks already decided the review status. Do not override it.
Write a concise, actionable review comment for the current phase.
Focus on evidence from artifacts and state fields. Mention retry actions only when status is retry or fatal.
"""

BACKGROUND_REVIEWER_SYSTEM_PROMPT = """
You are the Background Reviewer for a data-science LangGraph workflow.
The deterministic checks already decided the status for the Reader output. Do not override it.
Return a single JSON object with key "comment".
The comment should be concise and actionable.
"""


def build_reviewer_comment_prompt(
    *,
    phase: str,
    status: ReviewStatus,
    check_result: ReviewCheckResult,
) -> str:
    """Build the human-readable review comment prompt from hard-check evidence."""

    failures = "\n".join(f"- {item}" for item in check_result.blocking_failures) or "- None"
    warnings = "\n".join(f"- {item}" for item in check_result.warnings) or "- None"
    artifacts = "\n".join(f"- {item}" for item in check_result.checked_artifacts) or "- None"
    state_fields = "\n".join(f"- {item}" for item in check_result.checked_state_fields) or "- None"

    return f"""
Phase: {phase}
Deterministic review status: {status}

Blocking failures:
{failures}

Warnings:
{warnings}

Checked artifacts:
{artifacts}

Checked state fields:
{state_fields}

Write the final reviewer comment in 3 to 6 short bullet points.
"""


def build_background_reviewer_comment_prompt(
    *,
    status: ReviewStatus,
    check_result: ReviewCheckResult,
    fallback_comment: str,
) -> str:
    """Build the Background Reviewer comment prompt from hard-check evidence."""

    failures = "\n".join(f"- {item}" for item in check_result.blocking_failures) or "- None"
    warnings = "\n".join(f"- {item}" for item in check_result.warnings) or "- None"
    state_fields = "\n".join(f"- {item}" for item in check_result.checked_state_fields) or "- None"

    return f"""
Reviewed phase: reader
Deterministic review status: {status}

Blocking failures:
{failures}

Warnings:
{warnings}

Checked state fields:
{state_fields}

Fallback comment:
{fallback_comment}

Return JSON only:
{{
  "comment": "short reviewer comment"
}}
"""


__all__ = [
    "BACKGROUND_REVIEWER_SYSTEM_PROMPT",
    "REVIEWER_COMMENT_SYSTEM_PROMPT",
    "build_background_reviewer_comment_prompt",
    "build_reviewer_comment_prompt",
]
