"""Agent-level LLM configuration management.

Reads per-agent LLM settings from environment variables using the
``AGENT_<UPPER_NAME>_<FIELD>`` naming convention and returns a strongly-typed
:class:`AgentLLMConfig` Pydantic model.

Environment variable mapping (example for agent "reader")::

    AGENT_READER_MODEL         → model identifier, e.g. ``dashscope/glm-5``
    AGENT_READER_TEMPERATURE   → float, sampling randomness
    AGENT_READER_MAX_TOKENS    → int, hard token limit
    AGENT_READER_STREAM        → bool, enable streaming (unused by helpers)
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Default model when no AGENT_<NAME>_MODEL is set.
_DEFAULT_MODEL = "dashscope/glm-5"


class AgentLLMConfig(BaseModel):
    """LLM configuration for a single agent."""

    model: str = _DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 65536
    stream: bool = False


def get_agent_config(agent_name: str) -> AgentLLMConfig:
    """Return the LLM configuration for *agent_name*.

    Reads ``AGENT_<UPPER_NAME>_{MODEL,TEMPERATURE,MAX_TOKENS,STREAM}``
    from the environment.  Missing values fall back to defaults.
    Invalid values trigger a warning log and also fall back.
    """
    load_dotenv()

    prefix = f"AGENT_{agent_name.upper()}"

    model = os.environ.get(f"{prefix}_MODEL", "") or _DEFAULT_MODEL

    temperature = _parse_float(
        os.environ.get(f"{prefix}_TEMPERATURE", ""),
        default=0.0,
        field_name=f"{prefix}_TEMPERATURE",
    )

    max_tokens = _parse_int(
        os.environ.get(f"{prefix}_MAX_TOKENS", ""),
        default=65536,
        field_name=f"{prefix}_MAX_TOKENS",
    )

    stream = _parse_bool(
        os.environ.get(f"{prefix}_STREAM", ""),
        default=False,
    )

    return AgentLLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_float(raw: str, *, default: float, field_name: str) -> float:
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid float value for %s=%r, using default %s", field_name, raw, default
        )
        return default


def _parse_int(raw: str, *, default: int, field_name: str) -> int:
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid int value for %s=%r, using default %s", field_name, raw, default
        )
        return default


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _parse_bool(raw: str, *, default: bool) -> bool:
    if not raw:
        return default
    lower = raw.strip().lower()
    if lower in _BOOL_TRUE:
        return True
    if lower in _BOOL_FALSE:
        return False
    return default


__all__ = ["AgentLLMConfig", "get_agent_config"]
