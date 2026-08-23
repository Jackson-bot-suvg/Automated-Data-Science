"""Project configuration helpers backed by the repository `.env` file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"
FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def dotenv_config() -> dict[str, str]:
    """Load non-empty config values from the project `.env` file."""

    if not DOTENV_PATH.exists():
        return {}

    try:
        from dotenv import dotenv_values

        raw_values = dotenv_values(DOTENV_PATH)
        return {
            str(key): str(value).strip()
            for key, value in raw_values.items()
            if key and value is not None and str(value).strip() != ""
        }
    except Exception:
        return _parse_dotenv(DOTENV_PATH)


def config_value(name: str, default: str | None = None) -> str | None:
    """Return a config value, preferring explicit process overrides over `.env`."""

    process_value = os.getenv(name)
    if process_value is not None and process_value.strip() != "":
        return process_value.strip()
    return dotenv_config().get(name, default)


def config_bool(name: str, *, default: bool = False) -> bool:
    """Read a boolean config value from `.env`/process config."""

    raw = config_value(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    return values


__all__ = [
    "DOTENV_PATH",
    "PROJECT_ROOT",
    "config_bool",
    "config_value",
    "dotenv_config",
]
