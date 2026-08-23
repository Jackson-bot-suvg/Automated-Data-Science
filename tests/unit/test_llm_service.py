"""Unit tests for shared LLM service helpers (now in app.llm.helpers)."""

from __future__ import annotations

import asyncio

from app.llm.helpers import complete_json, parse_json_response


def test_parse_json_response_handles_fenced_json() -> None:
    result = parse_json_response(
        (
            "```json\n"
            '{"next_phase": "data_cleaning", '
            '"phase_objectives": {"data_cleaning": "clean data"}, '
            '"review_points": ["ok"]}\n'
            "```"
        ),
        fallback={"next_phase": "preliminary_eda"},
    )

    assert result["next_phase"] == "data_cleaning"
    assert result["phase_objectives"]["data_cleaning"] == "clean data"
    assert result["review_points"] == ["ok"]


def test_parse_json_response_returns_fallback_for_invalid_json() -> None:
    fallback = {"next_phase": "preliminary_eda"}

    result = parse_json_response("not json", fallback=fallback)

    assert result == fallback


def test_complete_json_returns_fallback_when_no_provider_configured(
    monkeypatch,
) -> None:
    # When LLMService has no valid provider, complete_json should return fallback.
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    fallback = {"next_phase": "preliminary_eda"}

    result = asyncio.get_event_loop().run_until_complete(
        complete_json(
            agent_name="planner",
            system_prompt="system",
            user_prompt="user",
            fallback=fallback,
        )
    )

    assert result == fallback
