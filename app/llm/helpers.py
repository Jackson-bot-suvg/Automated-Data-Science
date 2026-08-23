"""High-level LLM helper functions for agent use cases.

Provides two async helpers:

- :func:`complete_json` — call LLM and parse a JSON response (with fallback).
- :func:`render_text` — call LLM and return raw text (with fallback).

Both helpers resolve the agent's model configuration via
:func:`~app.llm.agent_config.get_agent_config` and delegate the actual
LLM call to :class:`~app.llm.service.LLMService.chat`.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from app.llm.agent_config import get_agent_config
from app.llm.schemas import ChatMessage, ChatResponse
from app.llm.service import LLMService

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


async def complete_json(
    *,
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    """Call the LLM for *agent_name* and parse a JSON object response.

    On any failure (LLM error, JSON parse error, non-dict result) the
    function returns a shallow copy of *fallback* without raising.
    """
    fallback_payload = dict(fallback)

    try:
        cfg = get_agent_config(agent_name)
        # Lazy import to avoid circular dependency with app.llm.__init__
        from app.llm import get_service as _get_service

        service = _get_service()

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        result = await service.chat(
            model=cfg.model,
            messages=messages,
            module_name=agent_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stream=False,
        )

        # stream=False guarantees a ChatResponse
        assert isinstance(result, ChatResponse)
        text = result.content

        return parse_json_response(text, fallback=fallback_payload)

    except Exception:
        logger.debug(
            "complete_json failed for agent=%s, returning fallback",
            agent_name,
            exc_info=True,
        )
        return fallback_payload


async def render_text(
    *,
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    fallback: str,
) -> str:
    """Call the LLM for *agent_name* and return the text response.

    On any failure (LLM error, empty content) the function returns
    *fallback* without raising.
    """
    try:
        cfg = get_agent_config(agent_name)
        # Lazy import to avoid circular dependency with app.llm.__init__
        from app.llm import get_service as _get_service

        service = _get_service()

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        result = await service.chat(
            model=cfg.model,
            messages=messages,
            module_name=agent_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stream=False,
        )

        assert isinstance(result, ChatResponse)
        text = result.content.strip()
        return text or fallback

    except Exception:
        logger.debug(
            "render_text failed for agent=%s, returning fallback",
            agent_name,
            exc_info=True,
        )
        return fallback


def parse_json_response(
    response_text: str, *, fallback: Mapping[str, Any]
) -> dict[str, Any]:
    """Parse plain or fenced JSON object text.

    Behaviour is identical to the legacy ``parse_json_response`` from
    ``app.services.llm_service`` so that callers see no functional change.
    """
    fallback_payload = dict(fallback)
    text = (response_text or "").strip()
    if not text:
        return fallback_payload

    # Extract fenced JSON block if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Find first JSON object in text.
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        text = object_match.group(0).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return fallback_payload

    if not isinstance(parsed, dict):
        return fallback_payload
    return parsed


__all__ = ["complete_json", "render_text", "parse_json_response"]
