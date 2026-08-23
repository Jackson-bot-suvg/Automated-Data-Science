from __future__ import annotations

from app.config import DOTENV_PATH, config_bool, config_value, dotenv_config


def test_project_dotenv_is_tracked_config_source() -> None:
    assert DOTENV_PATH.exists()
    assert dotenv_config()["LLM_MODEL"] == "glm-5"
    assert dotenv_config()["DASHSCOPE_BASE_URL"].endswith("/compatible-mode/v1")


def test_process_config_can_override_dotenv_for_tests(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")

    assert config_value("LLM_MODEL") == "test-model"


def test_config_bool_reads_dotenv_and_process_values(monkeypatch) -> None:
    assert config_bool("LLM_ENABLED") is True

    monkeypatch.setenv("LLM_ENABLED", "false")

    assert config_bool("LLM_ENABLED") is False
