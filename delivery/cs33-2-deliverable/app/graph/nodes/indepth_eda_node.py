from app.agents.indepth_eda import run_indepth_eda
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState

async def indepth_eda_node(state: GraphState) -> dict:
    return await timed_node(state, node_name="indepth_eda", runner=run_indepth_eda)
