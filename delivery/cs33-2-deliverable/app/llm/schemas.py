"""LLM service layer data schemas.

Defines unified data structures for cross-provider LLM interactions:
messages, responses, usage info, parameters, and errors.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message with role and content."""

    role: str  # "system" | "user" | "assistant"
    content: str


class UsageInfo(BaseModel):
    """Token usage information from an LLM response."""

    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    """Complete (non-streaming) LLM response."""

    content: str
    model: str
    usage: UsageInfo | None = None
    parsed: Any | None = None


class ChatParams(BaseModel):
    """Optional parameters for LLM chat calls.

    Attributes:
        response_schema: When set, the provider will attempt to parse the
            LLM response as this Pydantic model and populate
            ``ChatResponse.parsed``.  Forces non-streaming mode.
        max_tokens: Upper bound on the number of generated tokens.
    """

    temperature: float | None = None
    top_p: float | None = None
    stream: bool = True
    thinking_budget: int | None = None
    response_schema: type[BaseModel] | None = None
    max_tokens: int | None = None


class LLMError(Exception):
    """Unified error raised by the LLM service layer.

    Wraps provider-specific SDK exceptions with context about which
    provider and model caused the failure.
    """

    def __init__(self, provider: str, model: str, message: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}/{model}] {message}")
