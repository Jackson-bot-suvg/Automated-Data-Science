from app.agents.summarizer import run_technical_summary
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def summarizer_node(state: GraphState) -> dict:
    return await timed_node(state, node_name="summarizer", runner=run_technical_summary)
