"""Routing placeholders for review gates and retry transitions."""

from __future__ import annotations

from app.schemas.io_contracts import PHASE_SEQUENCE
from app.schemas import REVIEW_GATED_PHASES
from app.schemas.state_schema import GraphState


def route_after_planner(state: GraphState) -> str:
    phase = state.current_phase

    if phase not in PHASE_SEQUENCE:
        raise ValueError(
            f"Invalid phase: {phase}. "
            f"Expected phase set: {PHASE_SEQUENCE}"
        )
    return phase 

def route_after_reviewer(state: GraphState) -> str:
    if state.review_state.latest_review_status =="approved":
        if state.current_phase ==  "model_building":
            return "summarizer"
        else:
            return "approved"

    if state.review_state.latest_review_status =="fatal":
        return "aborted"
    return "retry"

def route_after_summarizer(state: GraphState) -> str:
    phase = state.current_phase

    if phase not in PHASE_SEQUENCE:
        raise ValueError(
            f"Invalid phase: {phase}. "
            f"Expected phase set: {PHASE_SEQUENCE}"
        )

    if phase in {"model_building", "technical_summary"}:
        return "final"
    return "next_phase"

__all__ = [
    "REVIEW_GATED_PHASES",
    "route_after_planner",
    "route_after_reviewer",
    "route_after_summarizer",
]
