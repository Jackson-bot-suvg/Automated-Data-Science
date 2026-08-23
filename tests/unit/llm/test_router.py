"""Tests for app.llm.router — model identifier parsing."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.llm.router import ModelId, parse_model_id


# ── Valid identifiers ────────────────────────────────────────────────


class TestParseModelIdValid:
    def test_two_segment_openai(self) -> None:
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=False):
            mid = parse_model_id("openai/GPT-5.4")
        assert mid == ModelId(provider="openai", prefix=None, model="GPT-5.4")

    def test_three_segment_with_prefix(self) -> None:
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=False):
            mid = parse_model_id("openai/zai/glm-5.1")
        assert mid == ModelId(provider="openai", prefix="zai", model="glm-5.1")

    def test_anthropic_format(self) -> None:
        env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
        with patch.dict(os.environ, env, clear=False):
            mid = parse_model_id("anthropic/claude-opus-4.6")
        assert mid == ModelId(
            provider="anthropic", prefix=None, model="claude-opus-4.6"
        )

    def test_anthropic_with_prefix(self) -> None:
        env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
        with patch.dict(os.environ, env, clear=False):
            mid = parse_model_id("anthropic/custom/claude-3")
        assert mid == ModelId(provider="anthropic", prefix="custom", model="claude-3")

    def test_dynamic_provider_from_unified_config(self) -> None:
        env = {
            "LLM_PROVIDERS": "dashscope",
            "LLM_DASHSCOPE_API_KEY": "sk-xxx",
        }
        with patch.dict(os.environ, env, clear=False):
            mid = parse_model_id("dashscope/qwen-plus")
        assert mid == ModelId(provider="dashscope", prefix=None, model="qwen-plus")


# ── Invalid identifiers ──────────────────────────────────────────────


class TestParseModelIdInvalid:
    def test_no_separator(self) -> None:
        with pytest.raises(ValueError, match="expected format"):
            parse_model_id("GPT-5.4")

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError):
            parse_model_id("")

    def test_unknown_provider_parses_successfully(self) -> None:
        """parse_model_id only validates format, not provider existence."""
        mid = parse_model_id("google/gemini-pro")
        assert mid.provider == "google"
        assert mid.model == "gemini-pro"
