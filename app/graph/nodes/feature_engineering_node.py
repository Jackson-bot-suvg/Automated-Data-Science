from app.agents.feature_engineering import run_feature_engineering
from app.graph.nodes._timed import timed_node
from app.schemas.state_schema import GraphState


async def feature_engineering_node(state: GraphState) -> dict:
    return await timed_node(
        state, node_name="feature_engineering", runner=run_feature_engineering
    )
