from app.agents.model_building import run_model_building
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def modeling_node(state: GraphState) -> dict:
    return await timed_node(state, node_name="model_building", runner=run_model_building)
