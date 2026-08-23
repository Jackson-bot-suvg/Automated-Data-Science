from __future__ import annotations

from app.schemas.state_schema import GraphState, TargetSpec


def test_graph_state_initializes_target_spec_defaults() -> None:
    state = GraphState()

    assert state.target_spec.status == "unresolved"
    assert state.target_spec.target_columns == []
    assert state.target_spec.target_mode == "unknown"
    assert state.target_spec.submission.submission_columns == []
    assert state.target_spec.evidence == []


def test_target_spec_single_target_semantics() -> None:
    target_spec = TargetSpec(
        status="resolved",
        target_columns=["Survived"],
        target_mode="single",
        task_family="classification",
        prediction_kind="label",
    )

    assert target_spec.status == "resolved"
    assert target_spec.target_mode == "single"
    assert target_spec.target_columns == ["Survived"]


def test_target_spec_multi_target_semantics() -> None:
    target_spec = TargetSpec(
        status="resolved",
        target_columns=["Defect_1", "Defect_2"],
        target_mode="multi",
        task_family="classification",
        prediction_kind="probability",
    )

    assert target_spec.status == "resolved"
    assert target_spec.target_mode == "multi"
    assert target_spec.target_columns == ["Defect_1", "Defect_2"]
