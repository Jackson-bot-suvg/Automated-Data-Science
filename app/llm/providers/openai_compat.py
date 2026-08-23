"""OpenAI-compatible LLM provider.

Uses the ``openai`` SDK's ``AsyncOpenAI`` client.  Supports any service
that exposes an OpenAI-compatible Chat Completions endpoint (DashScope,
ZhipuAI, local vLLM, etc.) via a custom ``base_url``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Union

import openai
from pydantic import BaseModel

from app.observability.langsmith import is_tracing_enabled
from app.llm.providers.base import BaseProvider, ChatResult
from app.llm.schemas import (
    ChatMessage,
    ChatParams,
    ChatResponse,
    LLMError,
    UsageInfo,
)

logger = logging.getLogger(__name__)


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert unified ChatMessage list to OpenAI-compatible dicts."""
    return [{"role": m.role, "content": m.content} for m in messages]


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_first_json_value(text: str) -> str | None:
    """Extract the first decodable JSON object/array from free-form text."""
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
            return text[idx : idx + end]
        except json.JSONDecodeError:
            continue
    return None


def _structured_json_candidates(raw_content: str) -> list[str]:
    """Return likely JSON payload candidates from a structured response."""
    candidates: list[str] = []

    def _add(value: str | None) -> None:
        if not value:
            return
        stripped = value.strip()
        if stripped and stripped not in candidates:
            candidates.append(stripped)

    _add(raw_content)
    for match in _FENCED_JSON_RE.finditer(raw_content):
        _add(match.group(1))

    without_think = _THINK_BLOCK_RE.sub("", raw_content)
    _add(without_think)

    # Try extracting the first JSON object from each candidate.
    for candidate in list(candidates):
        _add(_extract_first_json_value(candidate))

    return candidates


def _parse_structured_content(schema: type[BaseModel], raw_content: str) -> BaseModel:
    """Parse structured content with relaxed extraction for providers like MiniMax."""
    last_error: Exception | None = None
    for candidate in _structured_json_candidates(raw_content):
        try:
            return schema.model_validate_json(candidate)
        except Exception as exc:
            last_error = exc
            try:
                parsed_json = json.loads(candidate)
                return schema.model_validate(
                    _normalize_structured_payload(schema, parsed_json)
                )
            except Exception as inner_exc:
                last_error = inner_exc
                continue

    if last_error is None:
        last_error = ValueError("No JSON object could be extracted from response.")
    raise last_error


def _normalize_structured_payload(schema: type[BaseModel], payload: object) -> object:
    """Coerce provider-specific near-miss payloads into the requested schema shape."""
    field_names = set(getattr(schema, "model_fields", {}).keys())

    # MiniMax sometimes returns a single tool-call object or a bare list of steps
    # when the caller expects a ToolCallPlan.
    if field_names == {"plan_name", "steps"}:
        if isinstance(payload, dict):
            if {"tool", "parameters"}.issubset(payload.keys()):
                return {
                    "plan_name": "structured_fallback_autowrap",
                    "steps": [payload],
                }
            if "steps" in payload and "plan_name" not in payload:
                cloned = dict(payload)
                cloned["plan_name"] = "structured_fallback_autowrap"
                return cloned
        if isinstance(payload, list):
            return {
                "plan_name": "structured_fallback_autowrap",
                "steps": payload,
            }

    return payload


class OpenAICompatProvider(BaseProvider):
    """Provider for OpenAI-compatible Chat Completions endpoints."""

    def __init__(self, base_url: str, api_key: str, provider_name: str = "openai") -> None:
        if not api_key:
            raise ValueError("OpenAI-compatible provider requires an api_key")
        self._provider_name = provider_name
        client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._client = self._maybe_wrap_client_for_langsmith(client)

    def _maybe_wrap_client_for_langsmith(self, client: openai.AsyncOpenAI) -> openai.AsyncOpenAI:
        """Wrap OpenAI client so LangSmith can capture LLM token usage/cost."""
        if not is_tracing_enabled():
            return client
        try:
            from langsmith.wrappers import wrap_openai
        except ImportError:
            logger.warning(
                "LangSmith tracing is enabled but wrap_openai is unavailable; "
                "LLM usage/cost metrics may be missing from LangSmith dashboards."
            )
            return client
        return wrap_openai(client)

    async def chat(
        self,
        messages: list[ChatMessage],
        params: ChatParams,
        model_name: str,
        prefix: str | None = None,
    ) -> ChatResult:
        actual_model = f"{prefix}/{model_name}" if prefix else model_name
        oai_messages = _to_openai_messages(messages)

        kwargs: dict = {
            "model": actual_model,
            "messages": oai_messages,
        }
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.max_tokens is not None:
            kwargs["max_tokens"] = params.max_tokens
        # thinking_budget is silently ignored — OpenAI does not support it.

        # --- structured output path ---
        if params.response_schema is not None:
            return await self._structured_complete(
                kwargs=kwargs,
                schema=params.response_schema,
                oai_messages=oai_messages,
                provider=self._provider_name,
                model=actual_model,
            )

        if params.stream:
            return self._stream(kwargs, provider=self._provider_name, model=actual_model)
        return await self._complete(kwargs, provider=self._provider_name, model=actual_model)

    # ------------------------------------------------------------------
    # Fallback Helper
    # ------------------------------------------------------------------

    def _get_fallback_kwargs_list(self, kwargs: dict) -> list[dict]:
        """Generate a sequence of kwargs for model fallback if the primary model fails."""
        model = kwargs.get("model", "")
        # Only apply fallback for glm/qwen models
        if "glm" not in model and "qwen" not in model:
            return [kwargs]
            
        base_kwargs = {k: v for k, v in kwargs.items() if k not in ["model", "extra_body"]}
        
        return [
            {**base_kwargs, "model": "glm-5.1", "extra_body": {"enable_thinking": True}},
            {**base_kwargs, "model": "qwen3.6-max-preview", "extra_body": {"enable_thinking": True, "search_strategy": "agent_max"}},
            # Only inject tools if the original kwargs didn't have tools, to avoid overriding function calling
            {
                **base_kwargs, 
                "model": "qwen3.6-plus", 
                "extra_body": {"enable_thinking": True}, 
                "tools": base_kwargs.get("tools", [{"type": "web_search"}, {"type": "web_extractor"}, {"type": "code_interpreter"}])
            }
        ]

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def _complete(
        self, kwargs: dict, *, provider: str, model: str
    ) -> ChatResponse:
        fallback_list = self._get_fallback_kwargs_list(kwargs)
        last_exc = None

        for attempt_kwargs in fallback_list:
            attempt_model = attempt_kwargs.get("model", model)
            try:
                resp = await self._client.chat.completions.create(**attempt_kwargs)
                break  # Success
            except openai.OpenAIError as exc:
                logger.warning(f"Provider {provider} model {attempt_model} failed: {exc}. Trying next fallback if available.")
                last_exc = exc
        else:
            raise LLMError(provider=provider, model=model, message=f"All fallbacks failed. Last error: {last_exc}") from last_exc

        choice = resp.choices[0]
        usage = None
        if resp.usage:
            usage = UsageInfo(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
        return ChatResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Structured output (non-streaming)
    # ------------------------------------------------------------------

    async def _structured_complete(
        self,
        *,
        kwargs: dict,
        schema: type[BaseModel],
        oai_messages: list[dict[str, str]],
        provider: str,
        model: str,
    ) -> ChatResponse:
        """Use ``beta.chat.completions.parse()`` with a fallback to JSON mode."""
        try:
            resp = await self._client.beta.chat.completions.parse(
                **kwargs,
                response_format=schema,
            )
        except (openai.OpenAIError, Exception) as exc:
            logger.info("parse() not supported (%s), falling back to JSON mode", exc)
            return await self._structured_fallback(
                kwargs=kwargs,
                schema=schema,
                oai_messages=oai_messages,
                provider=provider,
                model=model,
                original_error=str(exc),
            )

        parsed_obj = resp.choices[0].message.parsed
        if parsed_obj is None:
            # The API returned a refusion — treat as parse failure.
            raw = resp.choices[0].message.content or ""
            raise LLMError(
                provider=provider,
                model=model,
                message=f"Structured output returned no parsed object. Raw content: {raw[:500]}",
            )

        usage = None
        if resp.usage:
            usage = UsageInfo(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
        return ChatResponse(
            content=resp.choices[0].message.content or "",
            model=resp.model,
            usage=usage,
            parsed=parsed_obj,
        )

    async def _structured_fallback(
        self,
        *,
        kwargs: dict,
        schema: type[BaseModel],
        oai_messages: list[dict[str, str]],
        provider: str,
        model: str,
        original_error: str,
    ) -> ChatResponse:
        """Fallback: JSON mode + schema-injected prompt + manual validation."""
        schema_json = schema.model_json_schema()
        schema_instruction = (
            "\n\nYou MUST respond with a JSON object that conforms to the "
            "following schema:\n"
            f"```json\n{json.dumps(schema_json, ensure_ascii=False)}\n```\n"
            "Do NOT wrap the JSON in markdown code fences."
        )
        injected_messages = list(oai_messages)
        # Append schema instruction to the last user message
        if injected_messages and injected_messages[-1]["role"] == "user":
            injected_messages[-1] = {
                "role": "user",
                "content": injected_messages[-1]["content"] + schema_instruction,
            }
        else:
            injected_messages.append({"role": "user", "content": schema_instruction})

        fallback_kwargs = {
            "model": kwargs["model"],
            "messages": injected_messages,
            "response_format": {"type": "json_object"},
        }
        if "temperature" in kwargs:
            fallback_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            fallback_kwargs["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            fallback_kwargs["max_tokens"] = kwargs["max_tokens"]

        fallback_list = self._get_fallback_kwargs_list(fallback_kwargs)
        last_exc = None

        for attempt_kwargs in fallback_list:
            attempt_model = attempt_kwargs.get("model", model)
            try:
                resp = await self._client.chat.completions.create(**attempt_kwargs)
                break
            except openai.OpenAIError as exc:
                logger.warning(f"Structured fallback for {attempt_model} failed: {exc}. Trying next if available.")
                last_exc = exc
        else:
            raise LLMError(provider=provider, model=model, message=f"All fallbacks failed. Last error: {last_exc}") from last_exc

        raw_content = resp.choices[0].message.content or ""
        try:
            parsed_obj = _parse_structured_content(schema, raw_content)
        except Exception as parse_exc:
            raise LLMError(
                provider=provider,
                model=model,
                message=(
                    f"Structured output fallback parse failed "
                    f"(original error: {original_error}): {parse_exc}. "
                    f"Raw content: {raw_content[:500]}"
                ),
            ) from parse_exc

        usage = None
        if resp.usage:
            usage = UsageInfo(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
        return ChatResponse(
            content=raw_content,
            model=resp.model,
            usage=usage,
            parsed=parsed_obj,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream(
        self, kwargs: dict, *, provider: str, model: str
    ) -> AsyncGenerator[str, None]:
        fallback_list = self._get_fallback_kwargs_list(kwargs)
        last_exc = None
        stream = None

        for attempt_kwargs in fallback_list:
            attempt_model = attempt_kwargs.get("model", model)
            attempt_kwargs["stream"] = True
            try:
                stream = await self._client.chat.completions.create(**attempt_kwargs)
                break
            except openai.OpenAIError as exc:
                logger.warning(f"Stream fallback for {attempt_model} failed: {exc}. Trying next if available.")
                last_exc = exc
        
        if not stream:
            raise LLMError(provider=provider, model=model, message=f"All fallbacks failed. Last error: {last_exc}") from last_exc

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
