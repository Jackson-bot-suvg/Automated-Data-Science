"""In-process token usage aggregation for local demo logging."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any


_lock = Lock()
_usage_by_module: dict[str, dict[str, Any]] = {}
_MAX_RECENT_INTERACTIONS = 5
_MAX_CONTENT_CHARS = 2000


def reset_token_usage() -> None:
    with _lock:
        _usage_by_module.clear()


def record_token_usage(
    *,
    module_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    model: str = "",
    input_messages: list[dict[str, str]] | None = None,
    output_content: str = "",
) -> None:
    module = (module_name or "unknown").strip().lower() or "unknown"
    in_tokens = int(input_tokens or 0)
    out_tokens = int(output_tokens or 0)
    total_tokens = in_tokens + out_tokens

    with _lock:
        bucket = _usage_by_module.setdefault(
            module,
            {
                "module": module,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "models": {},
                "recent_interactions": [],
            },
        )
        bucket["llm_calls"] += 1
        bucket["input_tokens"] += in_tokens
        bucket["output_tokens"] += out_tokens
        bucket["total_tokens"] += total_tokens
        if model:
            model_bucket = bucket["models"].setdefault(model, {"llm_calls": 0, "total_tokens": 0})
            model_bucket["llm_calls"] += 1
            model_bucket["total_tokens"] += total_tokens
        interactions = bucket["recent_interactions"]
        interactions.append(
            {
                "model": model,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": total_tokens,
                "input": _truncate_messages(input_messages or []),
                "output": _truncate_text(output_content),
            }
        )
        if len(interactions) > _MAX_RECENT_INTERACTIONS:
            del interactions[:-_MAX_RECENT_INTERACTIONS]


def snapshot_token_usage() -> dict[str, dict[str, Any]]:
    with _lock:
        return deepcopy(_usage_by_module)


def _truncate_text(text: str) -> str:
    if len(text) <= _MAX_CONTENT_CHARS:
        return text
    return f"{text[:_MAX_CONTENT_CHARS]}...[truncated]"


def _truncate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    trimmed: list[dict[str, str]] = []
    for msg in messages[-6:]:
        role = str(msg.get("role") or "")
        content = _truncate_text(str(msg.get("content") or ""))
        trimmed.append({"role": role, "content": content})
    return trimmed

