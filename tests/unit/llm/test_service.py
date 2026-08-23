"""Tests for app.llm.service — LLMService routing and params."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.schemas import (
    ChatMessage,
    ChatParams,
    ChatResponse,
    UsageInfo,
)
from app.llm.service import LLMService


def _make_mock_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.chat = AsyncMock()
    return provider


@pytest.fixture()
def openai_provider() -> AsyncMock:
    return _make_mock_provider()


@pytest.fixture()
def anthropic_provider() -> AsyncMock:
    return _make_mock_provider()


@pytest.fixture()
def service(openai_provider: AsyncMock, anthropic_provider: AsyncMock) -> LLMService:
    return LLMService(
        providers={"openai": openai_provider, "anthropic": anthropic_provider}
    )


# ── Routing ──────────────────────────────────────────────────────────


class TestRouting:
    @pytest.mark.asyncio
    async def test_routes_to_openai(
        self, service: LLMService, openai_provider: AsyncMock
    ) -> None:
        openai_provider.chat.return_value = ChatResponse(
            content="hi", model="glm-5", usage=None
        )
        msgs = [ChatMessage(role="user", content="hello")]
        await service.chat("openai/glm-5", msgs, stream=False)

        openai_provider.chat.assert_called_once()
        call_kwargs = openai_provider.chat.call_args
        assert call_kwargs.kwargs["model_name"] == "glm-5"
        assert call_kwargs.kwargs["prefix"] is None

    @pytest.mark.asyncio
    async def test_routes_to_anthropic(
        self, service: LLMService, anthropic_provider: AsyncMock
    ) -> None:
        anthropic_provider.chat.return_value = ChatResponse(
            content="hi", model="claude-opus-4.6", usage=None
        )
        msgs = [ChatMessage(role="user", content="hello")]
        await service.chat("anthropic/claude-opus-4.6", msgs, stream=False)

        anthropic_provider.chat.assert_called_once()
        call_kwargs = anthropic_provider.chat.call_args.kwargs
        assert call_kwargs["model_name"] == "claude-opus-4.6"
        assert call_kwargs["prefix"] is None

    @pytest.mark.asyncio
    async def test_routes_with_prefix(
        self, service: LLMService, openai_provider: AsyncMock
    ) -> None:
        openai_provider.chat.return_value = ChatResponse(
            content="ok", model="glm-5.1", usage=None
        )
        msgs = [ChatMessage(role="user", content="test")]
        await service.chat("openai/zai/glm-5.1", msgs, stream=False)

        call_kwargs = openai_provider.chat.call_args.kwargs
        assert call_kwargs["model_name"] == "glm-5.1"
        assert call_kwargs["prefix"] == "zai"


# ── Dynamic provider routing ─────────────────────────────────────────


class TestDynamicProviderRouting:
    @pytest.mark.asyncio
    async def test_routes_to_dynamic_provider(self) -> None:
        dashscope_provider = _make_mock_provider()
        dashscope_provider.chat.return_value = ChatResponse(
            content="hello from dashscope", model="qwen-plus", usage=None
        )
        service = LLMService(providers={"dashscope": dashscope_provider})
        msgs = [ChatMessage(role="user", content="hello")]
        await service.chat("dashscope/qwen-plus", msgs, stream=False)

        dashscope_provider.chat.assert_called_once()
        call_kwargs = dashscope_provider.chat.call_args.kwargs
        assert call_kwargs["model_name"] == "qwen-plus"

    @pytest.mark.asyncio
    async def test_routes_to_custom_anthropic_provider(self) -> None:
        claude_provider = _make_mock_provider()
        claude_provider.chat.return_value = ChatResponse(
            content="hello from claude", model="claude-opus-4.6", usage=None
        )
        service = LLMService(providers={"claude": claude_provider})
        msgs = [ChatMessage(role="user", content="hello")]
        await service.chat("claude/claude-opus-4.6", msgs, stream=False)

        claude_provider.chat.assert_called_once()


# ── Params ───────────────────────────────────────────────────────────


class TestParams:
    @pytest.mark.asyncio
    async def test_stream_defaults_to_true(
        self, service: LLMService, openai_provider: AsyncMock
    ) -> None:
        openai_provider.chat.return_value = ChatResponse(
            content="", model="m", usage=None
        )
        msgs = [ChatMessage(role="user", content="x")]
        await service.chat("openai/m", msgs)

        call_kwargs = openai_provider.chat.call_args.kwargs
        params: ChatParams = call_kwargs["params"]
        assert params.stream is True

    @pytest.mark.asyncio
    async def test_temperature_and_top_p_passed(
        self, service: LLMService, openai_provider: AsyncMock
    ) -> None:
        openai_provider.chat.return_value = ChatResponse(
            content="", model="m", usage=None
        )
        msgs = [ChatMessage(role="user", content="x")]
        await service.chat("openai/m", msgs, temperature=0.7, top_p=0.9)

        params: ChatParams = openai_provider.chat.call_args.kwargs["params"]
        assert params.temperature == 0.7
        assert params.top_p == 0.9

    @pytest.mark.asyncio
    async def test_thinking_budget_in_params(
        self, service: LLMService, openai_provider: AsyncMock
    ) -> None:
        openai_provider.chat.return_value = ChatResponse(
            content="", model="m", usage=None
        )
        msgs = [ChatMessage(role="user", content="x")]
        await service.chat("openai/m", msgs, thinking_budget=10000)

        params: ChatParams = openai_provider.chat.call_args.kwargs["params"]
        assert params.thinking_budget == 10000


# ── Error handling ───────────────────────────────────────────────────


class TestServiceErrors:
    @pytest.mark.asyncio
    async def test_invalid_model_id_raises_value_error(
        self, service: LLMService
    ) -> None:
        msgs = [ChatMessage(role="user", content="x")]
        with pytest.raises(ValueError, match="expected format"):
            await service.chat("no-slash", msgs)

    @pytest.mark.asyncio
    async def test_unregistered_provider_raises_value_error(self) -> None:
        service = LLMService(providers={})
        msgs = [ChatMessage(role="user", content="x")]
        with pytest.raises(ValueError, match="No provider registered"):
            await service.chat("openai/m", msgs)


# ── Integration: build providers from unified config ─────────────────


class TestBuildProvidersFromConfig:
    def test_build_from_unified_config(self) -> None:
        env = {
            "LLM_PROVIDERS": "dashscope,zhipuai",
            "LLM_DASHSCOPE_TYPE": "openai_compat",
            "LLM_DASHSCOPE_API_KEY": "sk-a",
            "LLM_DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "LLM_ZHIPUAI_TYPE": "openai_compat",
            "LLM_ZHIPUAI_API_KEY": "sk-b",
            "LLM_ZHIPUAI_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
        }
        with patch.dict(os.environ, env, clear=False):
            providers = LLMService._build_providers()

        assert "dashscope" in providers
        assert "zhipuai" in providers
        # Verify correct provider types are built
        from app.llm.providers.openai_compat import OpenAICompatProvider

        assert isinstance(providers["dashscope"], OpenAICompatProvider)
        assert isinstance(providers["zhipuai"], OpenAICompatProvider)

    def test_build_from_legacy_config(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("LLM_PROVIDERS", None)
            providers = LLMService._build_providers()

        assert "openai" in providers
        assert "anthropic" in providers

    def test_build_mixed_types_from_unified(self) -> None:
        env = {
            "LLM_PROVIDERS": "dashscope,claude",
            "LLM_DASHSCOPE_TYPE": "openai_compat",
            "LLM_DASHSCOPE_API_KEY": "sk-a",
            "LLM_DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "LLM_CLAUDE_TYPE": "anthropic",
            "LLM_CLAUDE_API_KEY": "sk-ant-xxx",
        }
        with patch.dict(os.environ, env, clear=False):
            providers = LLMService._build_providers()

        from app.llm.providers.openai_compat import OpenAICompatProvider
        from app.llm.providers.anthropic import AnthropicProvider

        assert isinstance(providers["dashscope"], OpenAICompatProvider)
        assert isinstance(providers["claude"], AnthropicProvider)
