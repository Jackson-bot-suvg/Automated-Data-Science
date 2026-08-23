"""LLM service layer — unified interface for multi-provider LLM calls.

Usage::

    from app.llm import get_service, ChatMessage

    service = get_service()
    resp = await service.chat("openai/glm-5", [
        ChatMessage(role="user", content="Hello"),
    ], stream=False)
    print(resp.content)
"""

from __future__ import annotations

from app.llm.agent_config import AgentLLMConfig, get_agent_config
from app.llm.helpers import complete_json, parse_json_response, render_text
from app.llm.schemas import (
    ChatMessage,
    ChatParams,
    ChatResponse,
    LLMError,
    UsageInfo,
)
from app.llm.service import LLMService

__all__ = [
    "AgentLLMConfig",
    "ChatMessage",
    "ChatParams",
    "ChatResponse",
    "LLMError",
    "LLMService",
    "UsageInfo",
    "complete_json",
    "get_agent_config",
    "get_service",
    "parse_json_response",
    "render_text",
]

_service: LLMService | None = None


def get_service() -> LLMService:
    """Return a module-level singleton ``LLMService`` instance."""
    global _service  # noqa: PLW0603
    if _service is None:
        _service = LLMService()
    return _service
