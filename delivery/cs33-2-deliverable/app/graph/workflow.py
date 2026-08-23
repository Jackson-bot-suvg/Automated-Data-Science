"""Workflow construction for the LangGraph entry point."""

from __future__ import annotations

from app.schemas import LANGGRAPH_RUNTIME_CONTRACT
from app.graph.nodes.bg_reviewer_node import bg_reviewer_node
from app.graph.nodes.data_cleaning_node import data_cleaning_node
from app.graph.nodes.feature_engineering_node import feature_engineering_node
from app.graph.nodes.indepth_eda_node import indepth_eda_node
from app.graph.nodes.modeling_node import modeling_node
from app.graph.nodes.prelim_eda_node import prelim_eda_node
from app.graph.nodes.planner_node import planner_node
from app.graph.nodes.reader_node import reader_node
from app.graph.nodes.reviewer_node import reviewer_node
from app.graph.nodes.summarizer_node import summarizer_node
from app.graph.routes import (
    route_after_planner,
    route_after_reviewer,
    route_after_summarizer,
)
from app.schemas.state_schema import GraphState

def build_workflow(*, stop_after_technical_summary: bool = False):
    """Build the LangGraph workflow.

    Args:
        stop_after_technical_summary: When true, stop after the technical
            summarizer produces the technical report instead of routing into
            the business translator. The default keeps the original full graph.
    """
    from langgraph.graph import END, START, StateGraph
    graph_builder = StateGraph(GraphState)

    #--------------------------------------------------------------------------
    # Node Registration
    #--------------------------------------------------------------------------

    graph_builder.add_node('reader', reader_node)
    graph_builder.add_node("background_reviewer", bg_reviewer_node)
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("preliminary_eda",prelim_eda_node)
    graph_builder.add_node("data_cleaning", data_cleaning_node)
    graph_builder.add_node("indepth_eda", indepth_eda_node)
    graph_builder.add_node("feature_engineering", feature_engineering_node)
    graph_builder.add_node("model_building", modeling_node)
    graph_builder.add_node("reviewer", reviewer_node)
    graph_builder.add_node("summarizer", summarizer_node)
    if not stop_after_technical_summary:
        from app.graph.nodes.business_translator_node import business_translator_node

        graph_builder.add_node("business_translator", business_translator_node)


    #--------------------------------------------------------------------------
    # Edge Registration
    #--------------------------------------------------------------------------
    graph_builder.add_edge(START, "reader")
    graph_builder.add_edge("reader", "background_reviewer")
    graph_builder.add_conditional_edges(
        "background_reviewer", 
        route_after_reviewer,
       {
            "approved" : "planner",  # graph enters execution chain directly
            "retry" : "reader",
            "aborted" : END,
        }
    )
    graph_builder.add_conditional_edges(
        "planner", 
        route_after_planner,
        {
            "preliminary_eda": "preliminary_eda",
            "data_cleaning": "data_cleaning",
            "indepth_eda": "indepth_eda",
            "feature_engineering": "feature_engineering",
            "model_building": "model_building"
        }
    ) 
    # all execution phases
    graph_builder.add_edge("preliminary_eda", "reviewer")  
    graph_builder.add_edge("data_cleaning", "reviewer")  
    graph_builder.add_edge("indepth_eda", "reviewer")  
    graph_builder.add_edge("feature_engineering", "reviewer")  
    graph_builder.add_edge("model_building", "reviewer")      
    graph_builder.add_conditional_edges(
        "reviewer", 
        route_after_reviewer,
        {
            "approved" : "planner",      # pass to planner of next phase
            "retry" : "planner",         # pass to planner of current phase
            "summarizer" : "summarizer", # pass to summarizer for model_building phase
            "aborted" : END,
        }
    )
    graph_builder.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {
            "final" : END if stop_after_technical_summary else "business_translator",
            "next_phase": "planner",  # non-final fallback entry
        }
        
    )
    if not stop_after_technical_summary:
        graph_builder.add_edge("business_translator", END)
    graph = graph_builder.compile()

    return graph



__all__ = ["LANGGRAPH_RUNTIME_CONTRACT", "build_workflow"]
