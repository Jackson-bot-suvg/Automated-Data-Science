"""LangSmith observability helpers.

This module centralizes tracing configuration so business nodes remain unaware of
LangSmith details. Tracing is opt-in and controlled via environment variables.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import config_bool, config_value
from app.schemas.state_schema import GraphState


@dataclass(frozen=True)
class LangSmithConfig:
    requested: bool
    enabled: bool
    project_name: str | None
    workspace_id: str | None
    api_key_present: bool
    app_env: str
    reason: str | None = None


def _env_flag(name: str, *, default: bool = False) -> bool:
    return config_bool(name, default=default)


def get_langsmith_config() -> LangSmithConfig:
    requested = _env_flag("LANGSMITH_TRACING", default=False)
    project_name = _clean_env(config_value("LANGSMITH_PROJECT"))
    workspace_id = _clean_env(config_value("LANGSMITH_WORKSPACE_ID"))
    api_key = _clean_env(config_value("LANGSMITH_API_KEY"))
    app_env = _clean_env(config_value("APP_ENV")) or "dev"

    if not requested:
        return LangSmithConfig(
            requested=False,
            enabled=False,
            project_name=project_name,
            workspace_id=workspace_id,
            api_key_present=bool(api_key),
            app_env=app_env,
            reason="LANGSMITH_TRACING is not enabled.",
        )

    if not api_key:
        return LangSmithConfig(
            requested=True,
            enabled=False,
            project_name=project_name,
            workspace_id=workspace_id,
            api_key_present=False,
            app_env=app_env,
            reason="LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing.",
        )

    try:
        import langsmith  # noqa: F401
    except ImportError:
        return LangSmithConfig(
            requested=True,
            enabled=False,
            project_name=project_name,
            workspace_id=workspace_id,
            api_key_present=True,
            app_env=app_env,
            reason="LANGSMITH_TRACING=true but langsmith is not installed.",
        )

    return LangSmithConfig(
        requested=True,
        enabled=True,
        project_name=project_name,
        workspace_id=workspace_id,
        api_key_present=True,
        app_env=app_env,
    )


def is_tracing_enabled() -> bool:
    return get_langsmith_config().enabled


def with_tracing_context(
    *,
    project_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> contextlib.AbstractContextManager[Any]:
    config = get_langsmith_config()
    if not config.enabled:
        if config.requested and config.reason:
            warnings.warn(config.reason, RuntimeWarning, stacklevel=2)
        return contextlib.nullcontext()

    import langsmith as ls

    context_kwargs: dict[str, Any] = {"enabled": True}
    chosen_project = project_name or config.project_name
    if chosen_project:
        context_kwargs["project_name"] = chosen_project
    if tags:
        context_kwargs["tags"] = list(tags)
    if metadata:
        context_kwargs["metadata"] = dict(metadata)

    return ls.tracing_context(**context_kwargs)


def without_tracing_context() -> contextlib.AbstractContextManager[Any]:
    """Temporarily disable LangSmith tracing within the enclosed scope."""
    config = get_langsmith_config()
    if not config.enabled:
        return contextlib.nullcontext()
    try:
        import langsmith as ls
    except ImportError:
        return contextlib.nullcontext()
    return ls.tracing_context(enabled=False)


async def run_with_root_trace(
    *,
    run_name: str,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    coro_fn: Any,
) -> Any:
    """Run an async function under a single explicit LangSmith root trace."""
    config = get_langsmith_config()
    if not config.enabled:
        return await coro_fn()
    try:
        from langsmith import traceable
    except ImportError:
        return await coro_fn()

    @traceable(run_type="chain", name=run_name, tags=list(tags or []), metadata=dict(metadata or {}))
    async def _wrapped() -> Any:
        return await coro_fn()

    return await _wrapped()


def get_current_run_url() -> str:
    """Return current LangSmith run URL if available."""
    if not is_tracing_enabled():
        return ""
    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:
        return ""

    run_tree = get_current_run_tree()
    if run_tree is None:
        return ""

    # RunTree exposes get_url() in recent SDKs.
    try:
        url = run_tree.get_url()
        return str(url or "")
    except Exception:
        pass

    # Fallback for potential SDK variations.
    url = getattr(run_tree, "url", "")
    return str(url or "")


def get_latest_run_url(*, thread_id: str = "", dataset: str = "") -> str:
    """Fallback: query LangSmith for the latest run URL."""
    config = get_langsmith_config()
    if not config.enabled or not config.project_name:
        return ""
    try:
        from langsmith import Client
    except ImportError:
        return ""

    try:
        client = Client()
        runs = client.list_runs(project_name=config.project_name, limit=20)
    except Exception:
        return ""

    selected = _select_latest_matching_run(runs, thread_id=thread_id, dataset=dataset)

    if selected is None:
        return ""

    root_run = _resolve_root_run(client, selected)
    try:
        return str(root_run.get_url() or "")
    except Exception:
        pass
    return str(getattr(root_run, "url", "") or "")


def get_current_public_run_url() -> str:
    """Return public share URL for current run if available."""
    if not is_tracing_enabled():
        return ""
    config = get_langsmith_config()
    if not config.project_name:
        return ""
    try:
        from langsmith import Client
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:
        return ""

    run_tree = get_current_run_tree()
    if run_tree is None:
        return ""

    run_id = _root_run_id_from_run(run_tree)
    if not run_id:
        return ""
    try:
        client = Client()
        return _share_run_and_get_public_url(client, run_id)
    except Exception:
        return ""


def get_latest_public_run_url(*, thread_id: str = "", dataset: str = "") -> str:
    """Fallback: locate latest run and return a public share URL."""
    config = get_langsmith_config()
    if not config.enabled or not config.project_name:
        return ""
    try:
        from langsmith import Client
    except ImportError:
        return ""

    try:
        client = Client()
        runs = client.list_runs(project_name=config.project_name, limit=20)
    except Exception:
        return ""

    selected = _select_latest_matching_run(runs, thread_id=thread_id, dataset=dataset)

    if selected is None:
        return ""

    root_run = _resolve_root_run(client, selected)
    run_id = str(getattr(root_run, "id", "") or "")
    if not run_id:
        return ""
    try:
        return _share_run_and_get_public_url(client, run_id)
    except Exception:
        return ""


def _share_run_and_get_public_url(client: Any, run_id: str) -> str:
    """Create/retrieve share token and format canonical public URL."""
    share_obj: Any = None
    # Preferred API on recent SDKs.
    try:
        share_obj = client.share_run(run_id)
    except Exception:
        share_obj = None

    # Some SDKs expose read_share_token helpers instead.
    if share_obj is None:
        for method_name in ("read_run_shared_link", "read_run_share_link", "read_run_shared_url"):
            method = getattr(client, method_name, None)
            if callable(method):
                try:
                    share_obj = method(run_id)
                    break
                except Exception:
                    continue

    # If share_run returns a URL string directly.
    if isinstance(share_obj, str):
        return share_obj

    share_token = ""
    if isinstance(share_obj, dict):
        share_token = str(
            share_obj.get("share_token")
            or share_obj.get("token")
            or share_obj.get("id")
            or ""
        )
        maybe_url = str(share_obj.get("url") or "")
        if maybe_url:
            return maybe_url
    else:
        share_token = str(
            getattr(share_obj, "share_token", "")
            or getattr(share_obj, "token", "")
            or getattr(share_obj, "id", "")
            or ""
        )
        maybe_url = str(getattr(share_obj, "url", "") or "")
        if maybe_url:
            return maybe_url

    if share_token:
        return f"https://smith.langchain.com/public/{share_token}/r"
    return ""


def _select_latest_matching_run(runs: Any, *, thread_id: str, dataset: str) -> Any:
    normalized_thread_id = str(thread_id or "").strip()
    normalized_dataset = str(dataset or "").strip()
    for run in runs:
        metadata = getattr(run, "metadata", None)
        run_thread_id = ""
        run_dataset = ""
        if isinstance(metadata, dict):
            run_thread_id = str(metadata.get("thread_id") or "").strip()
            run_dataset = str(metadata.get("dataset") or "").strip()
        # If caller asked for filters, enforce exact match.
        if normalized_thread_id and run_thread_id != normalized_thread_id:
            continue
        if normalized_dataset and run_dataset != normalized_dataset:
            continue
        return run
    return None


def _root_run_id_from_run(run: Any) -> str:
    trace_id = str(getattr(run, "trace_id", "") or "").strip()
    if trace_id:
        return trace_id
    return str(getattr(run, "id", "") or "").strip()


def _resolve_root_run(client: Any, run: Any) -> Any:
    root_id = _root_run_id_from_run(run)
    if root_id:
        try:
            return client.read_run(root_id)
        except Exception:
            pass
    return run


def get_langsmith_tracing_extra(
    *,
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
    }


def invoke_graph_with_tracing(
    graph: Any,
    state: GraphState,
    *,
    config: Mapping[str, Any] | None = None,
) -> Any:
    metadata = workflow_trace_metadata(state)
    tags = workflow_trace_tags(state)
    project_name = get_langsmith_config().project_name

    with with_tracing_context(project_name=project_name, tags=tags, metadata=metadata):
        invoke_kwargs: dict[str, Any] = {}
        if config:
            invoke_kwargs["config"] = dict(config)
        return graph.invoke(state, **invoke_kwargs)


def workflow_trace_metadata(state: GraphState) -> dict[str, Any]:
    dataset_name = infer_dataset_name(
        train_path=state.task_config.dataset_paths.train or state.data_state.current_train_path,
        test_path=state.task_config.dataset_paths.test or state.data_state.current_test_path,
    )
    return {
        "app_name": "26s1-cs33-2",
        "workflow": "multi_agent_ds",
        "thread_id": state.runtime_state.thread_id or "",
        "run_id": state.runtime_state.run_id or state.task_config.task_id or "",
        "current_phase": state.current_phase or "",
        "task_type": state.task_config.task_type or state.problem_understanding.predicted_task_type or "",
        "target_column": state.task_config.target_column or state.problem_understanding.target_column or "",
        "dataset": dataset_name or "",
        "train_path": _basename_or_empty(state.task_config.dataset_paths.train or state.data_state.current_train_path),
        "test_path": _basename_or_empty(state.task_config.dataset_paths.test or state.data_state.current_test_path),
    }


def workflow_trace_tags(state: GraphState) -> list[str]:
    config = get_langsmith_config()
    dataset_name = infer_dataset_name(
        train_path=state.task_config.dataset_paths.train or state.data_state.current_train_path,
        test_path=state.task_config.dataset_paths.test or state.data_state.current_test_path,
    )
    tags = ["langgraph", "planner", f"env:{config.app_env}"]
    if dataset_name:
        tags.append(f"dataset:{dataset_name}")
    return tags


def planner_trace_metadata(
    *,
    planner_type: str,
    profile_json: Mapping[str, Any],
    model_name: str,
    temperature: float,
    max_tokens: int,
    target_column: str = "",
) -> dict[str, Any]:
    dataset_shape = _safe_mapping(profile_json.get("dataset_shape"))
    normalized_shape: dict[str, Any] = {}
    if dataset_shape:
        normalized_shape = {
            "rows": dataset_shape.get("rows"),
            "columns": dataset_shape.get("columns"),
        }

    resolved_target = target_column or str(profile_json.get("target_variable") or "")

    return {
        "ls_provider": "dashscope",
        "ls_model_name": model_name,
        "planner_type": planner_type,
        "phase": planner_type,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "target_column": resolved_target,
        "dataset_shape": normalized_shape,
    }


def planner_trace_tags(*, planner_type: str) -> list[str]:
    return [
        "planner",
        "llm",
        f"phase:{planner_type}",
    ]


def infer_dataset_name(*, train_path: str, test_path: str) -> str:
    for raw_path in (train_path, test_path):
        path = _clean_env(raw_path)
        if not path:
            continue
        parts = Path(path).parts
        if "titanic" in {part.lower() for part in parts}:
            return "titanic"
        parent = Path(path).parent.name.strip().lower()
        if parent and parent not in {".", "data", "artifacts"}:
            return parent
    return ""


def _basename_or_empty(path: str) -> str:
    raw = _clean_env(path)
    if not raw:
        return ""
    return Path(raw).name


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


__all__ = [
    "get_current_public_run_url",
    "get_current_run_url",
    "get_latest_public_run_url",
    "get_latest_run_url",
    "LangSmithConfig",
    "get_langsmith_config",
    "get_langsmith_tracing_extra",
    "infer_dataset_name",
    "invoke_graph_with_tracing",
    "is_tracing_enabled",
    "planner_trace_metadata",
    "planner_trace_tags",
    "run_with_root_trace",
    "with_tracing_context",
    "without_tracing_context",
    "workflow_trace_metadata",
    "workflow_trace_tags",
]
