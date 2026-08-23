"""LLM provider configuration management.

Loads API credentials and endpoint URLs from environment variables.
Uses python-dotenv to ensure .env file values are available via os.environ.
Configuration priority: system env vars > .env file > code defaults.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Valid provider names: letters, digits, hyphens, underscores only.
_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Supported provider types.
ProviderType = Literal["openai_compat", "anthropic"]
_SUPPORTED_TYPES: set[str] = {"openai_compat", "anthropic"}


class OpenAICompatConfig(BaseModel):
    """Configuration for an OpenAI-compatible provider."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""


class AnthropicConfig(BaseModel):
    """Configuration for the Anthropic provider."""

    base_url: str = "https://api.anthropic.com"
    api_key: str = ""


class ProviderConfig(BaseModel):
    """Aggregated configuration for all LLM providers."""

    providers: dict[str, OpenAICompatConfig | AnthropicConfig]


def _validate_provider_name(name: str) -> None:
    """Raise ValueError if *name* contains characters outside [a-zA-Z0-9_-]."""
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"Invalid provider name '{name}': "
            f"only letters, digits, hyphens and underscores are allowed."
        )


def _load_from_legacy() -> ProviderConfig:
    """Load configuration using the legacy environment variable format.

    Reads ``OPENAI_API_KEY``/``OPENAI_BASE_URL`` and
    ``ANTHROPIC_API_KEY``/``ANTHROPIC_BASE_URL``.
    """
    openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    openai_cfg = OpenAICompatConfig(
        base_url=openai_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=openai_api_key,
    )
    anthropic_cfg = AnthropicConfig(
        base_url=anthropic_base_url or AnthropicConfig.model_fields["base_url"].default,
        api_key=anthropic_api_key,
    )

    return ProviderConfig(
        providers={
            "openai": openai_cfg,
            "anthropic": anthropic_cfg,
        }
    )


def _load_from_unified(providers_env: str) -> ProviderConfig:
    """Load configuration using the unified ``LLM_<NAME>_*`` format.

    Args:
        providers_env: Comma-separated list of provider names from the
            ``LLM_PROVIDERS`` environment variable.

    Returns:
        A ``ProviderConfig`` with one entry per declared provider.

    Raises:
        ValueError: If a provider name is invalid or an unsupported type is
            specified.
    """
    providers: dict[str, OpenAICompatConfig | AnthropicConfig] = {}

    for raw_name in providers_env.split(","):
        name = raw_name.strip()
        if not name:
            continue

        _validate_provider_name(name)

        upper = name.upper()
        p_type = os.environ.get(f"LLM_{upper}_TYPE", "openai_compat").strip()

        if p_type not in _SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported provider type '{p_type}' for provider '{name}'. "
                f"Supported types: {sorted(_SUPPORTED_TYPES)}"
            )

        api_key = os.environ.get(f"LLM_{upper}_API_KEY", "")
        base_url = os.environ.get(f"LLM_{upper}_BASE_URL", "")

        if p_type == "anthropic":
            cfg: OpenAICompatConfig | AnthropicConfig = AnthropicConfig(
                base_url=base_url or AnthropicConfig.model_fields["base_url"].default,
                api_key=api_key,
            )
        else:
            # openai_compat
            cfg = OpenAICompatConfig(
                base_url=base_url
                or OpenAICompatConfig.model_fields["base_url"].default,
                api_key=api_key,
            )

        if not api_key:
            logger.warning(
                "Provider '%s' has no API key configured (LLM_%s_API_KEY is empty). "
                "Calls using this provider will fail at runtime.",
                name,
                upper,
            )

        providers[name] = cfg

    return ProviderConfig(providers=providers)


def load_provider_config() -> ProviderConfig:
    """Load provider configuration from environment variables.

    Calls ``load_dotenv()`` first so that values from ``.env`` are available
    via ``os.environ``.  System env vars take precedence over ``.env`` values.

    When ``LLM_PROVIDERS`` is set, uses the unified format
    (``LLM_<NAME>_TYPE`` / ``LLM_<NAME>_API_KEY`` / ``LLM_<NAME>_BASE_URL``).
    Otherwise falls back to the legacy ``OPENAI_*`` / ``ANTHROPIC_*`` variables.
    """
    provider_env_keys = {
        "LLM_PROVIDERS",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    }
    if not any(key in os.environ for key in provider_env_keys):
        load_dotenv()

    providers_env = os.environ.get("LLM_PROVIDERS", "").strip()
    if providers_env:
        logger.info(
            "Using unified provider configuration (LLM_PROVIDERS=%s)", providers_env
        )
        return _load_from_unified(providers_env)

    logger.info("LLM_PROVIDERS not set, falling back to legacy configuration format")
    return _load_from_legacy()
