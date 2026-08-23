"""Anthropic LLM provider.

Uses the ``anthropic`` SDK's ``AsyncAnthropic`` client.
Handles system-message separation and thinking_budget passthrough.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Union

import anthropic
from pydantic import BaseModel

from app.llm.providers.base import BaseProvider, ChatResult
from app.llm.schemas import (
    ChatMessage,
    ChatParams,
    ChatResponse,
    LLMError,
    UsageInfo,
)

logger = logging.getLogger(__name__)


def _separate_system(
    messages: list[ChatMessage],
) -> tuple[str | None, list[dict[str, str]]]:
    """Extract system messages and return (system_text, non_system_messages).

    Anthropic requires the system prompt as a separate parameter.
    Multiple system messages are joined with newlines.
    """
    system_parts: list[str] = []
    non_system: list[dict[str, str]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            non_system.append({"role": m.role, "content": m.content})
    system_text = "\n".join(system_parts) if system_parts else None
    return system_text, non_system


class AnthropicProvider(BaseProvider):
    """Provider for the Anthropic Messages API."""

    def __init__(self, base_url: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("Anthropic provider requires an api_key")
        self._client = anthropic.AsyncAnthropic(base_url=base_url, api_key=api_key)

    async def chat(
        self,
        messages: list[ChatMessage],
        params: ChatParams,
        model_name: str,
        prefix: str | None = None,
    ) -> ChatResult:
        # prefix is ignored for Anthropic — not applicable.
        system_text, non_system = _separate_system(messages)

        # --- structured output: reject streaming ---
        if params.response_schema is not None and params.stream:
            raise ValueError(
                "Structured output is only supported in non-streaming mode. "
                "Set stream=False when using response_schema."
            )

        kwargs: dict = {
            "model": model_name,
            "messages": non_system,
        }
        if system_text:
            kwargs["system"] = system_text
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.max_tokens is not None:
            kwargs["max_tokens"] = params.max_tokens
        if params.thinking_budget is not None:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": params.thinking_budget,
            }

        # --- structured output path ---
        if params.response_schema is not None:
            return await self._structured_complete(
                kwargs=kwargs,
                schema=params.response_schema,
                model=model_name,
            )

        if params.stream:
            return self._stream(kwargs, model=model_name)
        return await self._complete(kwargs, model=model_name)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def _complete(self, kwargs: dict, *, model: str) -> ChatResponse:
        try:
            resp = await self._client.messages.create(**kwargs)
        except anthropic.AnthropicError as exc:
            raise LLMError(provider="anthropic", model=model, message=str(exc)) from exc

        # resp.content is a list of content blocks; concatenate text blocks.
        text = "".join(block.text for block in resp.content if block.type == "text")
        usage = None
        if resp.usage:
            usage = UsageInfo(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
        return ChatResponse(content=text, model=resp.model, usage=usage)

    # ------------------------------------------------------------------
    # Structured output (non-streaming)
    # ------------------------------------------------------------------

    async def _structured_complete(
        self,
        *,
        kwargs: dict,
        schema: type[BaseModel],
        model: str,
    ) -> ChatResponse:
        """Inject JSON schema into system prompt, enable JSON mode, parse response."""
        schema_json = schema.model_json_schema()
        schema_instruction = (
            "\n\nYou MUST respond with a JSON object that conforms to the "
            "following schema:\n"
            f"```json\n{json.dumps(schema_json, ensure_ascii=False)}\n```\n"
            "Do NOT wrap the JSON in markdown code fences. Output ONLY the JSON object."
        )

        # Inject schema constraint into system prompt
        structured_kwargs = dict(kwargs)
        current_system = structured_kwargs.get("system", "")
        structured_kwargs["system"] = (
            current_system + schema_instruction
            if current_system
            else schema_instruction.strip()
        )

        # Ensure max_tokens is set (Anthropic requires it)
        if "max_tokens" not in structured_kwargs:
            structured_kwargs["max_tokens"] = 8192

        try:
            resp = await self._client.messages.create(**structured_kwargs)
        except anthropic.AnthropicError as exc:
            raise LLMError(provider="anthropic", model=model, message=str(exc)) from exc

        text = "".join(block.text for block in resp.content if block.type == "text")

        try:
            parsed_obj = schema.model_validate_json(text)
        except Exception as parse_exc:
            raise LLMError(
                provider="anthropic",
                model=model,
                message=(
                    f"Structured output parse failed: {parse_exc}. "
                    f"Raw content: {text[:500]}"
                ),
            ) from parse_exc

        usage = None
        if resp.usage:
            usage = UsageInfo(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
        return ChatResponse(
            content=text,
            model=resp.model,
            usage=usage,
            parsed=parsed_obj,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream(self, kwargs: dict, *, model: str) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AnthropicError as exc:
            raise LLMError(provider="anthropic", model=model, message=str(exc)) from exc
