"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Union

from app.llm.schemas import ChatMessage, ChatParams, ChatResponse

# Type alias for the return value of chat methods.
ChatResult = Union[ChatResponse, AsyncGenerator[str, None]]


class BaseProvider(ABC):
    """Interface that every LLM provider must implement."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        params: ChatParams,
        model_name: str,
        prefix: str | None = None,
    ) -> ChatResult:
        """Send messages and return either a complete response or an async token stream.

        Args:
            messages: Chat history.
            params: Optional parameters.

                - **temperature**, **top_p**: Sampling controls.
                - **stream**: When *True* (default) the method returns an async
                  token generator; otherwise a :class:`ChatResponse`.
                - **thinking_budget**: Extended-thinking token budget (Anthropic
                  only).
                - **response_schema**: When set, the provider parses the LLM
                  output into this Pydantic model and populates
                  ``ChatResponse.parsed``.  Providers MUST force non-streaming
                  mode when this field is present.
                - **max_tokens**: Hard limit on generated tokens.  Passed
                  directly to the underlying API when set.

            model_name: The model name without provider prefix.
            prefix: Optional path prefix for compatible services.

        Returns:
            ``ChatResponse`` when ``params.stream`` is False,
            or ``AsyncGenerator[str, None]`` when ``params.stream`` is True.
        """
        ...
