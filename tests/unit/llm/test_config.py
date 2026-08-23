"""Tests for app.llm.config — provider configuration loading."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.llm.config import (
    AnthropicConfig,
    OpenAICompatConfig,
    _load_from_legacy,
    _load_from_unified,
    _validate_provider_name,
    load_provider_config,
)


# ── Legacy format tests ─────────────────────────────────────────────


class TestLoadFromLegacy:
    def test_reads_openai_env_vars(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-test-123",
            "OPENAI_BASE_URL": "https://api.example.com/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_legacy()

        openai_cfg = cfg.providers["openai"]
        assert openai_cfg.api_key == "sk-test-123"
        assert openai_cfg.base_url == "https://api.example.com/v1"

    def test_reads_anthropic_env_vars(self) -> None:
        env = {
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_legacy()

        anthropic_cfg = cfg.providers["anthropic"]
        assert anthropic_cfg.api_key == "sk-ant-test"
        assert "anthropic.com" in anthropic_cfg.base_url

    def test_openai_default_base_url_when_empty(self) -> None:
        env = {"OPENAI_API_KEY": "sk-123"}
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_legacy()

        assert (
            cfg.providers["openai"].base_url
            == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def test_missing_keys_give_empty_strings(self) -> None:
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            for key in [
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
            ]:
                os.environ.pop(key, None)
            cfg = _load_from_legacy()

        assert cfg.providers["openai"].api_key == ""
        assert cfg.providers["anthropic"].api_key == ""

    def test_both_providers_registered(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            cfg = _load_from_legacy()

        assert "openai" in cfg.providers
        assert "anthropic" in cfg.providers

    def test_system_env_overrides_dotenv(self) -> None:
        """load_dotenv does not overwrite existing env vars by default."""
        env = {"OPENAI_API_KEY": "sk-system"}
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_legacy()

        assert cfg.providers["openai"].api_key == "sk-system"


# ── Unified format tests ────────────────────────────────────────────


class TestLoadFromUnified:
    def test_single_openai_compat_provider(self) -> None:
        env = {
            "LLM_PROVIDERS": "dashscope",
            "LLM_DASHSCOPE_TYPE": "openai_compat",
            "LLM_DASHSCOPE_API_KEY": "sk-dash",
            "LLM_DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert "dashscope" in cfg.providers
        ds = cfg.providers["dashscope"]
        assert isinstance(ds, OpenAICompatConfig)
        assert ds.api_key == "sk-dash"
        assert ds.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_multiple_openai_compat_providers(self) -> None:
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
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert len(cfg.providers) == 2
        assert "dashscope" in cfg.providers
        assert "zhipuai" in cfg.providers
        assert cfg.providers["dashscope"].api_key == "sk-a"
        assert cfg.providers["zhipuai"].api_key == "sk-b"

    def test_mixed_types(self) -> None:
        env = {
            "LLM_PROVIDERS": "dashscope,anthropic",
            "LLM_DASHSCOPE_TYPE": "openai_compat",
            "LLM_DASHSCOPE_API_KEY": "sk-a",
            "LLM_DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "LLM_ANTHROPIC_TYPE": "anthropic",
            "LLM_ANTHROPIC_API_KEY": "sk-ant-xxx",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert isinstance(cfg.providers["dashscope"], OpenAICompatConfig)
        assert isinstance(cfg.providers["anthropic"], AnthropicConfig)

    def test_default_type_is_openai_compat(self) -> None:
        env = {
            "LLM_PROVIDERS": "myprovider",
            "LLM_MYPROVIDER_API_KEY": "sk-xxx",
            "LLM_MYPROVIDER_BASE_URL": "https://api.example.com/v1",
        }
        # No LLM_MYPROVIDER_TYPE set → should default to openai_compat
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert isinstance(cfg.providers["myprovider"], OpenAICompatConfig)

    def test_default_base_url_for_openai_compat(self) -> None:
        env = {
            "LLM_PROVIDERS": "myprovider",
            "LLM_MYPROVIDER_API_KEY": "sk-xxx",
            # No BASE_URL set
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert cfg.providers["myprovider"].base_url == "https://api.openai.com/v1"

    def test_default_base_url_for_anthropic(self) -> None:
        env = {
            "LLM_PROVIDERS": "claude",
            "LLM_CLAUDE_TYPE": "anthropic",
            "LLM_CLAUDE_API_KEY": "sk-ant-xxx",
            # No BASE_URL set
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert cfg.providers["claude"].base_url == "https://api.anthropic.com"

    def test_unsupported_type_raises_value_error(self) -> None:
        env = {
            "LLM_PROVIDERS": "myprovider",
            "LLM_MYPROVIDER_TYPE": "gemini",
            "LLM_MYPROVIDER_API_KEY": "sk-xxx",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="Unsupported provider type 'gemini'"):
                _load_from_unified(env["LLM_PROVIDERS"])

    def test_invalid_provider_name_raises_value_error(self) -> None:
        env = {
            "LLM_PROVIDERS": "bad name!",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="Invalid provider name"):
                _load_from_unified(env["LLM_PROVIDERS"])

    def test_name_with_special_chars_raises_value_error(self) -> None:
        env = {
            "LLM_PROVIDERS": "my.provider",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="Invalid provider name"):
                _load_from_unified(env["LLM_PROVIDERS"])

    def test_valid_names_with_hyphens_and_underscores(self) -> None:
        env = {
            "LLM_PROVIDERS": "my-provider,another_provider",
            "LLM_MY_PROVIDER_API_KEY": "sk-a",
            "LLM_ANOTHER_PROVIDER_API_KEY": "sk-b",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert "my-provider" in cfg.providers
        assert "another_provider" in cfg.providers

    def test_case_insensitive_env_lookup(self) -> None:
        env = {
            "LLM_PROVIDERS": "MyProvider",
            "LLM_MYPROVIDER_API_KEY": "sk-xxx",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert "MyProvider" in cfg.providers
        assert cfg.providers["MyProvider"].api_key == "sk-xxx"

    def test_missing_api_key_logs_warning(self) -> None:
        env = {
            "LLM_PROVIDERS": "nokey",
            "LLM_NOKEY_BASE_URL": "https://api.example.com/v1",
            # No API_KEY set
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("app.llm.config.logger") as mock_logger:
                cfg = _load_from_unified(env["LLM_PROVIDERS"])

        mock_logger.warning.assert_called_once()
        assert "no API key" in mock_logger.warning.call_args[0][0]
        assert cfg.providers["nokey"].api_key == ""

    def test_empty_entries_skipped(self) -> None:
        env = {
            "LLM_PROVIDERS": "dashscope,,zhipuai",
            "LLM_DASHSCOPE_API_KEY": "sk-a",
            "LLM_ZHIPUAI_API_KEY": "sk-b",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = _load_from_unified(env["LLM_PROVIDERS"])

        assert len(cfg.providers) == 2


# ── load_provider_config dispatch tests ──────────────────────────────


class TestLoadProviderConfigDispatch:
    def test_falls_back_to_legacy_when_llm_providers_not_set(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-test-123",
            "OPENAI_BASE_URL": "https://api.example.com/v1",
        }
        # Ensure LLM_PROVIDERS is not set
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("LLM_PROVIDERS", None)
            cfg = load_provider_config()

        assert "openai" in cfg.providers
        assert "anthropic" in cfg.providers

    def test_uses_unified_when_llm_providers_set(self) -> None:
        env = {
            "LLM_PROVIDERS": "dashscope",
            "LLM_DASHSCOPE_API_KEY": "sk-xxx",
            "LLM_DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_provider_config()

        assert "dashscope" in cfg.providers
        assert "openai" not in cfg.providers

    def test_uses_unified_when_llm_providers_empty_string(self) -> None:
        """Empty LLM_PROVIDERS string falls back to legacy."""
        env = {
            "LLM_PROVIDERS": "",
            "OPENAI_API_KEY": "sk-test",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_provider_config()

        assert "openai" in cfg.providers


# ── Provider name validation tests ──────────────────────────────────


class TestValidateProviderName:
    def test_valid_simple_name(self) -> None:
        _validate_provider_name("dashscope")  # should not raise

    def test_valid_with_hyphen(self) -> None:
        _validate_provider_name("my-provider")

    def test_valid_with_underscore(self) -> None:
        _validate_provider_name("my_provider")

    def test_valid_with_digits(self) -> None:
        _validate_provider_name("provider123")

    def test_invalid_with_dot(self) -> None:
        with pytest.raises(ValueError, match="Invalid provider name"):
            _validate_provider_name("my.provider")

    def test_invalid_with_space(self) -> None:
        with pytest.raises(ValueError, match="Invalid provider name"):
            _validate_provider_name("my provider")

    def test_invalid_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid provider name"):
            _validate_provider_name("")
