"""Serializable contracts for reviewer checks and history records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReviewStatus = Literal["approved", "retry", "fatal"]


class ReviewCheckResult(BaseModel):
    """Deterministic reviewer tool output before LLM comment generation."""

    phase: str
    blocking_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_artifacts: list[str] = Field(default_factory=list)
    checked_state_fields: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.blocking_failures


class ReviewRecord(BaseModel):
    """Append-only entry for review_state.review_history."""

    phase: str
    status: ReviewStatus
    attempt: int
    blocking_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_artifacts: list[str] = Field(default_factory=list)
    checked_state_fields: list[str] = Field(default_factory=list)
    comment: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

