"""Reviewer phase entrypoint used by LangGraph."""

from __future__ import annotations

from app.agents.reviewer_agent import ReviewerAgent
from app.schemas.state_schema import GraphState


async def run_reviewer(state: GraphState) -> dict:
    """Execute the reviewer phase and return partial state updates."""

    return await ReviewerAgent().run(state)
