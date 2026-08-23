"""Tests for app.llm.providers — OpenAI and Anthropic provider implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.schemas import (
    ChatMessage,
    ChatParams,
    ChatResponse,
    LLMError,
)
from app.llm.providers.openai_compat import OpenAICompatProvider
from app.llm.providers.anthropic import AnthropicProvider


# ── OpenAI Compat Provider ───────────────────────────────────────────


class TestOpenAICompatProvider:
    def test_constructor_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            OpenAICompatProvider(base_url="https://api.example.com", api_key="")

    @pytest.mark.asyncio
    async def test_non_streaming_returns_chat_response(self) -> None:
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "glm-5"
        mock_resp.usage = mock_usage

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="hi")]
            params = ChatParams(stream=False)
            result = await provider.chat(msgs, params, model_name="glm-5")

        assert isinstance(result, ChatResponse)
        assert result.content == "Hello world"
        assert result.model == "glm-5"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_non_streaming_with_prefix(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "zai/glm-5.1"
        mock_resp.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="hi")]
            params = ChatParams(stream=False)
            result = await provider.chat(
                msgs, params, model_name="glm-5.1", prefix="zai"
            )

        assert isinstance(result, ChatResponse)
        # Verify the model kwarg includes the prefix
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "zai/glm-5.1"

    @pytest.mark.asyncio
    async def test_sdk_error_converted_to_llm_error(self) -> None:
        import openai

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(),
                body=None,
            )
        )

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="hi")]
            params = ChatParams(stream=False)
            with pytest.raises(LLMError, match="Invalid API key"):
                await provider.chat(msgs, params, model_name="glm-5")

    @pytest.mark.asyncio
    async def test_thinking_budget_ignored(self) -> None:
        """OpenAI provider should silently ignore thinking_budget."""
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "m"
        mock_resp.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="hi")]
            params = ChatParams(stream=False, thinking_budget=10000)
            await provider.chat(msgs, params, model_name="m")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "thinking_budget" not in call_kwargs


# ── Anthropic Provider ───────────────────────────────────────────────


class TestAnthropicProvider:
    def test_constructor_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            AnthropicProvider(base_url="https://api.anthropic.com", api_key="")

    @pytest.mark.asyncio
    async def test_system_message_separation(self) -> None:
        """System messages should be extracted into the 'system' parameter."""
        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 25

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello from Claude"

        mock_resp = MagicMock()
        mock_resp.content = [mock_text_block]
        mock_resp.model = "claude-opus-4.6"
        mock_resp.usage = mock_usage

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.anthropic.anthropic.AsyncAnthropic",
            return_value=mock_client,
        ):
            provider = AnthropicProvider(
                base_url="https://api.anthropic.com", api_key="sk-ant-test"
            )
            msgs = [
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="hi"),
            ]
            params = ChatParams(stream=False)
            result = await provider.chat(msgs, params, model_name="claude-opus-4.6")

        assert isinstance(result, ChatResponse)
        assert result.content == "Hello from Claude"

        # Verify system was passed separately
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"
        # Messages should not contain system
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    @pytest.mark.asyncio
    async def test_thinking_budget_passed(self) -> None:
        """Anthropic provider should pass thinking_budget as 'thinking' param."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "thinking..."

        mock_resp = MagicMock()
        mock_resp.content = [mock_text_block]
        mock_resp.model = "claude-opus-4.6"
        mock_resp.usage = MagicMock(input_tokens=10, output_tokens=20)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.anthropic.anthropic.AsyncAnthropic",
            return_value=mock_client,
        ):
            provider = AnthropicProvider(
                base_url="https://api.anthropic.com", api_key="sk-ant-test"
            )
            msgs = [ChatMessage(role="user", content="think hard")]
            params = ChatParams(stream=False, thinking_budget=10000)
            await provider.chat(msgs, params, model_name="claude-opus-4.6")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }

    @pytest.mark.asyncio
    async def test_sdk_error_converted_to_llm_error(self) -> None:
        import anthropic

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="invalid x-api-key",
                response=MagicMock(),
                body=None,
            )
        )

        with patch(
            "app.llm.providers.anthropic.anthropic.AsyncAnthropic",
            return_value=mock_client,
        ):
            provider = AnthropicProvider(
                base_url="https://api.anthropic.com", api_key="sk-ant-test"
            )
            msgs = [ChatMessage(role="user", content="hi")]
            params = ChatParams(stream=False)
            with pytest.raises(LLMError, match="invalid x-api-key"):
                await provider.chat(msgs, params, model_name="claude-opus-4.6")

    @pytest.mark.asyncio
    async def test_prefix_is_ignored(self) -> None:
        """Prefix is not applicable for Anthropic and should be ignored."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "ok"

        mock_resp = MagicMock()
        mock_resp.content = [mock_text_block]
        mock_resp.model = "claude-opus-4.6"
        mock_resp.usage = None

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.anthropic.anthropic.AsyncAnthropic",
            return_value=mock_client,
        ):
            provider = AnthropicProvider(
                base_url="https://api.anthropic.com", api_key="sk-ant-test"
            )
            msgs = [ChatMessage(role="user", content="hi")]
            params = ChatParams(stream=False)
            await provider.chat(
                msgs, params, model_name="claude-opus-4.6", prefix="ignored"
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4.6"
