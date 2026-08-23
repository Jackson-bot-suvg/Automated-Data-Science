from app.agents.reviewer import run_reviewer
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def reviewer_node(state: GraphState) -> dict:
    return await timed_node(state, node_name="reviewer", runner=run_reviewer)
