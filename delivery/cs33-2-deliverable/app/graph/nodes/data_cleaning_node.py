from app.agents.data_cleaning import run_data_cleaning
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def data_cleaning_node(state: GraphState) -> dict:
    return await timed_node(state, node_name="data_cleaning", runner=run_data_cleaning)
