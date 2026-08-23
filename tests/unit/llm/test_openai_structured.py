"""Tests for OpenAI-compatible provider structured output feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.llm.schemas import (
    ChatMessage,
    ChatParams,
    ChatResponse,
    LLMError,
)
from app.llm.providers.openai_compat import OpenAICompatProvider
from app.schemas.tools import ToolCallPlan


# 用于测试的简单 Pydantic 模型
class SimplePlan(BaseModel):
    """测试用的简单计划模型。"""

    action: str
    value: int


class TestOpenAIStructuredOutput:
    """测试 OpenAI 兼容提供商的结构化输出功能。"""

    @pytest.mark.asyncio
    async def test_structured_output_parse_succeeds(self) -> None:
        """测试结构化输出使用 parse 方法成功返回解析后的对象。"""
        # 创建模拟的响应对象
        mock_parsed_obj = SimplePlan(action="test", value=42)

        mock_message = MagicMock()
        mock_message.parsed = mock_parsed_obj
        mock_message.content = '{"action": "test", "value": 42}'

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "glm-5"
        mock_resp.usage = mock_usage

        # 创建模拟的客户端
        mock_client = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="生成一个计划")]
            params = ChatParams(response_schema=SimplePlan, stream=False)
            result = await provider.chat(msgs, params, model_name="glm-5")

        # 验证结果
        assert isinstance(result, ChatResponse)
        assert result.content == '{"action": "test", "value": 42}'
        assert result.model == "glm-5"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

        # 验证解析后的对象
        assert isinstance(result.parsed, SimplePlan)
        assert result.parsed.action == "test"
        assert result.parsed.value == 42

        # 验证调用了 parse 方法
        mock_client.beta.chat.completions.parse.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_output_falls_back_to_json_mode(self) -> None:
        """测试当 parse 方法失败时，自动回退到 JSON mode。"""
        import openai

        # 创建模拟的客户端
        mock_client = AsyncMock()

        # parse 方法抛出错误
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=openai.OpenAIError("not supported")
        )

        # 创建 fallback 的响应对象
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "fallback", "value": 99}'

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "glm-5"
        mock_resp.usage = None

        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="生成一个计划")]
            params = ChatParams(response_schema=SimplePlan, stream=False)
            result = await provider.chat(msgs, params, model_name="glm-5")

        # 验证结果
        assert isinstance(result, ChatResponse)
        assert result.content == '{"action": "fallback", "value": 99}'
        assert result.model == "glm-5"
        assert result.usage is None

        # 验证解析后的对象
        assert isinstance(result.parsed, SimplePlan)
        assert result.parsed.action == "fallback"
        assert result.parsed.value == 99

        # 验证调用了 parse 和 fallback 方法
        mock_client.beta.chat.completions.parse.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()

        # 验证 fallback 调用使用了 response_format={"type": "json_object"}
        fallback_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert fallback_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_structured_output_parse_failure_raises_llm_error(self) -> None:
        """测试当 parse 和 fallback 都失败时，抛出 LLMError。"""
        import openai

        # 创建模拟的客户端
        mock_client = AsyncMock()

        # parse 方法抛出错误
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=openai.OpenAIError("not supported")
        )

        # fallback 返回无效的 JSON
        mock_choice = MagicMock()
        mock_choice.message.content = "not json"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="生成一个计划")]
            params = ChatParams(response_schema=SimplePlan, stream=False)

            # 验证抛出 LLMError
            with pytest.raises(
                LLMError, match="Structured output fallback parse failed"
            ):
                await provider.chat(msgs, params, model_name="glm-5")

    @pytest.mark.asyncio
    async def test_structured_output_fallback_extracts_json_after_think_block(self) -> None:
        """兼容 MiniMax 一类在 JSON 前输出 <think> 文本的回复。"""
        import openai

        mock_client = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=openai.OpenAIError("not supported")
        )

        mock_choice = MagicMock()
        mock_choice.message.content = (
            "<think>first reason step by step</think>\n"
            '{"action":"fallback_after_think","value":7}'
        )

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "MiniMax-M2.7"
        mock_resp.usage = None

        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="生成一个计划")]
            params = ChatParams(response_schema=SimplePlan, stream=False)
            result = await provider.chat(msgs, params, model_name="MiniMax-M2.7")

        assert isinstance(result.parsed, SimplePlan)
        assert result.parsed.action == "fallback_after_think"
        assert result.parsed.value == 7

    @pytest.mark.asyncio
    async def test_structured_output_fallback_extracts_fenced_json_from_text(self) -> None:
        """兼容先解释再给 fenced JSON 的结构化回复。"""
        import openai

        mock_client = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=openai.OpenAIError("not supported")
        )

        mock_choice = MagicMock()
        mock_choice.message.content = (
            "Here is the plan.\n"
            "```json\n"
            '{"action":"fenced","value":11}\n'
            "```"
        )

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "MiniMax-M2.7"
        mock_resp.usage = None

        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="生成一个计划")]
            params = ChatParams(response_schema=SimplePlan, stream=False)
            result = await provider.chat(msgs, params, model_name="MiniMax-M2.7")

        assert isinstance(result.parsed, SimplePlan)
        assert result.parsed.action == "fenced"
        assert result.parsed.value == 11

    @pytest.mark.asyncio
    async def test_structured_output_fallback_wraps_single_tool_call_into_plan(self) -> None:
        """兼容 MiniMax 返回单个 step，而不是完整 ToolCallPlan。"""
        import openai

        mock_client = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=openai.OpenAIError("not supported")
        )

        mock_choice = MagicMock()
        mock_choice.message.content = (
            "<think>tool planning</think>\n"
            '{"tool":"encode_categorical","parameters":{"column":"Sex","method":"label"},"reasoning":"encode binary sex"}'
        )

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "MiniMax-M2.7"
        mock_resp.usage = None

        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="生成一个特征工程计划")]
            params = ChatParams(response_schema=ToolCallPlan, stream=False)
            result = await provider.chat(msgs, params, model_name="MiniMax-M2.7")

        assert isinstance(result.parsed, ToolCallPlan)
        assert result.parsed.plan_name == "structured_fallback_autowrap"
        assert len(result.parsed.steps) == 1
        assert result.parsed.steps[0].tool == "encode_categorical"

    @pytest.mark.asyncio
    async def test_max_tokens_passed_to_parse(self) -> None:
        """测试 max_tokens 参数正确传递给 parse 方法。"""
        # 创建模拟的响应对象
        mock_parsed_obj = SimplePlan(action="test", value=42)

        mock_message = MagicMock()
        mock_message.parsed = mock_parsed_obj
        mock_message.content = '{"action": "test", "value": 42}'

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "glm-5"
        mock_resp.usage = None

        # 创建模拟的客户端
        mock_client = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="生成一个计划")]
            params = ChatParams(
                response_schema=SimplePlan, max_tokens=65536, stream=False
            )
            await provider.chat(msgs, params, model_name="glm-5")

        # 验证 max_tokens 参数传递
        call_kwargs = mock_client.beta.chat.completions.parse.call_args.kwargs
        assert call_kwargs["max_tokens"] == 65536

    @pytest.mark.asyncio
    async def test_structured_forces_non_streaming(self) -> None:
        """测试即使设置 stream=True，结构化输出也会强制使用非流式模式。"""
        # 创建模拟的响应对象
        mock_parsed_obj = SimplePlan(action="test", value=42)

        mock_message = MagicMock()
        mock_message.parsed = mock_parsed_obj
        mock_message.content = '{"action": "test", "value": 42}'

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "glm-5"
        mock_resp.usage = None

        # 创建模拟的客户端
        mock_client = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_resp)

        with patch(
            "app.llm.providers.openai_compat.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAICompatProvider(
                base_url="https://api.example.com", api_key="sk-test"
            )
            msgs = [ChatMessage(role="user", content="生成一个计划")]
            # 即使设置 stream=True，response_schema 也会强制使用非流式模式
            params = ChatParams(
                response_schema=SimplePlan, stream=True, max_tokens=1024
            )
            result = await provider.chat(msgs, params, model_name="glm-5")

        # 验证返回的是 ChatResponse 而不是流
        assert isinstance(result, ChatResponse)
        assert isinstance(result.parsed, SimplePlan)

        # 验证调用了 parse 方法而不是流式方法
        mock_client.beta.chat.completions.parse.assert_called_once()
        # 确保没有调用流式方法
        assert not hasattr(mock_client.chat.completions.create, "stream") or (
            not mock_client.chat.completions.create.called
            or mock_client.chat.completions.create.call_args.kwargs.get("stream")
            is not True
        )
