from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorType(str, Enum):
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    SCHEMA_ERROR = "schema_error"
    ARTIFACT_MISSING = "artifact_missing"
    VALIDATION_FAILURE = "validation_failure"
    DATA_LEAKAGE_RISK = "data_leakage_risk"


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetails(StrictBaseModel):
    code: str | None = None
    step: str | None = None
    source: str | None = None
    artifact_key: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    traceback: str | None = None
    suggestions: list[str] = Field(default_factory=list)


class StandardError(StrictBaseModel):
    status: str = "error"
    phase: str = "unknown"
    error_type: ErrorType = ErrorType.RUNTIME_ERROR
    message: str
    details: ErrorDetails = Field(default_factory=ErrorDetails)
