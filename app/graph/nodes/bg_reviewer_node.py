from app.agents.bg_reviewer import run_bg_reviewer
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def bg_reviewer_node(state: GraphState) -> dict:
    return await timed_node(state, node_name="background_reviewer", runner=run_bg_reviewer)
