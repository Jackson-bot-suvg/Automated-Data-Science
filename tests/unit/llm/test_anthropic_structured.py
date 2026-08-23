"""Tests for Anthropic provider's structured output feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.llm.schemas import ChatMessage, ChatParams, ChatResponse, LLMError
from app.llm.providers.anthropic import AnthropicProvider


# 用于测试的简单 Pydantic 模型
class SimplePlan(BaseModel):
    """简单的计划模型，用于测试结构化输出。"""

    action: str
    value: int


class TestAnthropicStructuredOutput:
    """Anthropic provider 结构化输出测试。"""

    @pytest.mark.asyncio
    async def test_structured_output_succeeds(self) -> None:
        """验证结构化输出成功解析为 Pydantic 模型实例。"""
        # Mock Anthropic API 响应
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"action": "test", "value": 42}'

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
            msgs = [ChatMessage(role="user", content="Generate a plan")]
            params = ChatParams(response_schema=SimplePlan, stream=False)
            response = await provider.chat(msgs, params, model_name="claude-opus-4.6")

        # 验证返回的是 ChatResponse（非流式）
        assert isinstance(response, ChatResponse)
        # 验证响应的 parsed 字段是 SimplePlan 实例，且值正确
        assert response.parsed is not None
        assert isinstance(response.parsed, SimplePlan)
        assert response.parsed.action == "test"
        assert response.parsed.value == 42
        assert response.content == '{"action": "test", "value": 42}'
        assert response.model == "claude-opus-4.6"

    @pytest.mark.asyncio
    async def test_structured_output_injects_schema_in_system_prompt(self) -> None:
        """验证 JSON schema 被注入到 system prompt 中。"""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"action": "jump", "value": 100}'

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
            msgs = [ChatMessage(role="user", content="Generate a plan")]
            params = ChatParams(response_schema=SimplePlan, stream=False)
            await provider.chat(msgs, params, model_name="claude-opus-4.6")

        # 验证 system prompt 包含 schema 描述文本
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert "JSON object that conforms to" in call_kwargs["system"]
        assert '"action"' in call_kwargs["system"]
        assert '"value"' in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_structured_output_with_stream_raises_value_error(self) -> None:
        """验证结构化输出在 streaming 模式下抛出 ValueError。"""
        mock_client = AsyncMock()

        with patch(
            "app.llm.providers.anthropic.anthropic.AsyncAnthropic",
            return_value=mock_client,
        ):
            provider = AnthropicProvider(
                base_url="https://api.anthropic.com", api_key="sk-ant-test"
            )
            msgs = [ChatMessage(role="user", content="Generate a plan")]
            # stream=True 应该与 response_schema 不兼容
            params = ChatParams(response_schema=SimplePlan, stream=True)

            with pytest.raises(
                ValueError,
                match="Structured output is only supported in non-streaming mode",
            ):
                await provider.chat(msgs, params, model_name="claude-opus-4.6")

        # 验证没有调用 create 方法
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_structured_output_parse_failure_raises_llm_error(self) -> None:
        """验证解析失败时抛出 LLMError。"""
        # Mock 返回无效的 JSON
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "invalid json"

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
            msgs = [ChatMessage(role="user", content="Generate a plan")]
            params = ChatParams(response_schema=SimplePlan, stream=False)

            with pytest.raises(LLMError, match="parse failed"):
                await provider.chat(msgs, params, model_name="claude-opus-4.6")

    @pytest.mark.asyncio
    async def test_max_tokens_passed_to_create(self) -> None:
        """验证显式指定的 max_tokens 被传递给 API 调用。"""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"action": "test", "value": 42}'

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
            msgs = [ChatMessage(role="user", content="Generate a plan")]
            params = ChatParams(
                response_schema=SimplePlan, max_tokens=65536, stream=False
            )
            await provider.chat(msgs, params, model_name="claude-opus-4.6")

        # 验证 max_tokens 参数被传递
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 65536

    @pytest.mark.asyncio
    async def test_default_max_tokens_when_not_specified(self) -> None:
        """验证未指定 max_tokens 时使用默认值 8192。"""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"action": "test", "value": 42}'

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
            msgs = [ChatMessage(role="user", content="Generate a plan")]
            # 不指定 max_tokens
            params = ChatParams(response_schema=SimplePlan, stream=False)
            await provider.chat(msgs, params, model_name="claude-opus-4.6")

        # 验证使用默认 max_tokens 值
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 8192
