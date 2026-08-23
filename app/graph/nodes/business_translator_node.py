from app.agents.business_translator import run_business_translation
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def business_translator_node(state: GraphState) -> dict:
    async def _runner(inner_state: GraphState) -> dict:
        return run_business_translation(inner_state)

    return await timed_node(state, node_name="business_translation", runner=_runner)
