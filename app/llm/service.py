"""LLM service — unified entry point for all LLM interactions.

Routes model identifiers to the appropriate provider and exposes a
single ``chat()`` method that supports both streaming and non-streaming
responses.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Union

from app.llm.config import (
    AnthropicConfig,
    OpenAICompatConfig,
    load_provider_config,
)
from app.llm.providers.base import BaseProvider, ChatResult
from app.llm.router import ModelId, parse_model_id
from app.llm.schemas import (
    ChatMessage,
    ChatParams,
    ChatResponse,
    LLMError,
)
from app.observability.token_usage import record_token_usage


class LLMService:
    """Unified LLM service that dispatches requests to registered providers.

    Usage::

        service = LLMService()
        # Streaming (default)
        async for token in await service.chat("openai/glm-5", messages):
            print(token, end="")
        # Non-、
        resp = await service.chat("openai/glm-5", messages, stream=False)
    """

    def __init__(self, providers: dict[str, BaseProvider] | None = None) -> None:
        """Initialise the service.

        If *providers* is ``None`` the service is built from environment
        configuration via ``load_provider_config()``.
        """
        if providers is not None:
            self._providers: dict[str, BaseProvider] = providers
        else:
            self._providers = self._build_providers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResult:
        """Send a chat request routed by model identifier.

        Args:
            model: Model identifier, e.g. ``openai/glm-5`` or
                ``anthropic/claude-opus-4.6``.
            messages: Conversation messages.
            **kwargs: Optional overrides for ``ChatParams`` fields:

                - ``temperature`` / ``top_p``: Sampling controls.
                - ``stream``: *True* (default) for token generator,
                  *False* for :class:`ChatResponse`.
                - ``thinking_budget``: Extended-thinking token budget
                  (Anthropic only).
                - ``response_schema``: A Pydantic model class.  When set,
                  the provider parses the LLM output into this model
                  and populates ``ChatResponse.parsed``.  Forces
                  non-streaming mode.
                - ``max_tokens``: Hard limit on generated tokens.

        Returns:
            ``ChatResponse`` when ``stream=False``,
            ``AsyncGenerator[str, None]`` when ``stream=True`` (default).

        Raises:
            ValueError: If the model identifier is invalid.
            LLMError: If the provider call fails.
        """
        module_name = str(kwargs.pop("module_name", "") or "unknown")
        model_id = parse_model_id(model)
        provider = self._get_provider(model_id.provider)
        params = ChatParams(**kwargs)
        result = await provider.chat(
            messages=messages,
            params=params,
            model_name=model_id.model,
            prefix=model_id.prefix,
        )
        if isinstance(result, ChatResponse) and result.usage is not None:
            record_token_usage(
                module_name=module_name,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                model=result.model,
                input_messages=[{"role": m.role, "content": m.content} for m in messages],
                output_content=result.content or "",
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_provider(self, provider_name: str) -> BaseProvider:
        try:
            return self._providers[provider_name]
        except KeyError:
            raise ValueError(
                f"No provider registered for '{provider_name}'. "
                f"Available: {sorted(self._providers)}"
            ) from None

    @staticmethod
    def _build_providers() -> dict[str, BaseProvider]:
        cfg = load_provider_config()
        providers: dict[str, BaseProvider] = {}
        for name, pcfg in cfg.providers.items():
            if isinstance(pcfg, OpenAICompatConfig):
                from app.llm.providers.openai_compat import OpenAICompatProvider

                providers[name] = OpenAICompatProvider(
                    base_url=pcfg.base_url,
                    api_key=pcfg.api_key,
                    provider_name=name,
                )
            elif isinstance(pcfg, AnthropicConfig):
                try:
                    from app.llm.providers.anthropic import AnthropicProvider
                except ImportError as exc:
                    raise RuntimeError(
                        "Anthropic provider is configured but the 'anthropic' "
                        "package is not installed. Install project dependencies "
                        "or remove the anthropic provider from LLM_PROVIDERS."
                    ) from exc

                providers[name] = AnthropicProvider(
                    base_url=pcfg.base_url, api_key=pcfg.api_key
                )
        return providers
