from app.agents.reader import run_reader
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def reader_node(state: GraphState) -> dict:
    return await timed_node(state, node_name="reader", runner=run_reader)
