"""Business node entry points, exporting all LangGraph node functions.

Uses lazy imports so that importing only ``app.agents`` does not trigger
initialization of heavy optional dependencies such as ``matplotlib``.
This improves compatibility for tests and utility scripts.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["run_preliminary_eda", "run_data_cleaning", "run_feature_engineering"]

_EXPORT_TO_MODULE = {
    "run_preliminary_eda": "app.agents.preliminary_eda",
    "run_data_cleaning": "app.agents.data_cleaning",
    "run_feature_engineering": "app.agents.feature_engineering",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORT_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path)
    return getattr(module, name)
