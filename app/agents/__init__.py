"""业务节点入口，包含所有 LangGraph 节点函数的导出。

采用惰性导入，避免仅导入 `app.agents` 时触发重型可选依赖初始化
（例如 `matplotlib`），从而提升测试和工具脚本的兼容性。
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
