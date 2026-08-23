from app.graph.nodes import NODE_NAMES
from app.graph.workflow import build_workflow
from app.schemas import LANGGRAPH_RUNTIME_CONTRACT


def test_architecture_scaffold_exports() -> None:
    assert "reader" in NODE_NAMES
    assert "business_translation" in NODE_NAMES
    assert callable(build_workflow)
    assert LANGGRAPH_RUNTIME_CONTRACT["builder"] == "StateGraph(GraphState)"
