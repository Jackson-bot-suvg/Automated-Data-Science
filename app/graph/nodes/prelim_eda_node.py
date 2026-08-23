from app.agents.preliminary_eda import run_preliminary_eda
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def prelim_eda_node(state: GraphState) -> dict:
    async def _runner(inner_state: GraphState) -> dict:
        return run_preliminary_eda(inner_state)

    return await timed_node(state, node_name="preliminary_eda", runner=_runner)
