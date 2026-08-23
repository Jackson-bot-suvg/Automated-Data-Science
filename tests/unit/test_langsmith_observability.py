from __future__ import annotations

from contextlib import nullcontext

from app.observability.langsmith import (
    get_langsmith_config,
    invoke_graph_with_tracing,
    planner_trace_metadata,
    with_tracing_context,
    workflow_trace_metadata,
    workflow_trace_tags,
)
from app.schemas.state_schema import GraphState


def test_langsmith_tracing_disabled_is_noop(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    config = get_langsmith_config()

    assert config.enabled is False
    assert config.requested is False
    assert with_tracing_context().__class__ is nullcontext


def test_langsmith_requested_without_api_key_degrades_safely(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    config = get_langsmith_config()

    assert config.requested is True
    assert config.enabled is False
    assert "LANGSMITH_API_KEY" in (config.reason or "")


def test_workflow_metadata_and_tags_include_dataset_context(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")

    state = GraphState(
        current_phase="reader",
        task_config={
            "task_type": "classification",
            "target_column": "Survived",
            "dataset_paths": {
                "train": "data/titanic/train.csv",
                "test": "data/titanic/test.csv",
            },
        },
        runtime_state={"thread_id": "thread-123"},
    )

    metadata = workflow_trace_metadata(state)
    tags = workflow_trace_tags(state)

    assert metadata["app_name"] == "26s1-cs33-2"
    assert metadata["workflow"] == "multi_agent_ds"
    assert metadata["thread_id"] == "thread-123"
    assert metadata["dataset"] == "titanic"
    assert "langgraph" in tags
    assert "planner" in tags
    assert "dataset:titanic" in tags
    assert "env:test" in tags


def test_planner_metadata_extracts_shape_and_target() -> None:
    metadata = planner_trace_metadata(
        planner_type="data_cleaning",
        profile_json={
            "target_variable": "Survived",
            "dataset_shape": {"rows": 891, "columns": 12},
        },
        model_name="glm-5",
        temperature=1.0,
        max_tokens=65536,
    )

    assert metadata["planner_type"] == "data_cleaning"
    assert metadata["phase"] == "data_cleaning"
    assert metadata["target_column"] == "Survived"
    assert metadata["dataset_shape"] == {"rows": 891, "columns": 12}


def test_invoke_graph_with_tracing_preserves_graph_behavior(monkeypatch) -> None:
    entered = {"count": 0}

    class _FakeContext:
        def __enter__(self):
            entered["count"] += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeGraph:
        def __init__(self):
            self.calls = []

        def invoke(self, state, **kwargs):
            self.calls.append((state, kwargs))
            return {"ok": True}

    monkeypatch.setattr(
        "app.observability.langsmith.with_tracing_context",
        lambda **_: _FakeContext(),
    )

    state = GraphState(
        task_config={
            "dataset_paths": {
                "train": "data/titanic/train.csv",
                "test": "data/titanic/test.csv",
            }
        }
    )
    graph = _FakeGraph()

    result = invoke_graph_with_tracing(graph, state, config={"configurable": {"thread_id": "demo"}})

    assert result == {"ok": True}
    assert entered["count"] == 1
    assert graph.calls[0][1] == {"config": {"configurable": {"thread_id": "demo"}}}
