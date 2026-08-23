"""Helpers for updating artifact mappings in state updates."""

from __future__ import annotations

from typing import Any


def register_artifact(
    artifacts: dict[str, Any], *, key: str, path: str
) -> dict[str, Any]:
    """Return a copied artifact map with one artifact path updated."""

    updated = dict(artifacts or {})
    updated[str(key)] = str(path)
    return updated
