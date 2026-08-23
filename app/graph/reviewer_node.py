"""Graph-facing adapter for the reviewer agent."""

from __future__ import annotations

from app.agents.reviewer_agent import ReviewerAgent
from app.schemas.state_schema import GraphState


async def reviewer_node(state: GraphState) -> dict:
    """Run ReviewerAgent from a LangGraph node callback."""

    return await ReviewerAgent().run(state)


__all__ = ["reviewer_node"]
