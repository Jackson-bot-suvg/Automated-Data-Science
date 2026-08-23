from app.agents.planner import run_planner
from app.graph.nodes._timed import timed_node
from app.schemas.io_contracts import PHASE_SEQUENCE
from app.schemas.state_schema import GraphState


EXECUTION_PHASES: tuple[str, ...] = (
    "preliminary_eda",
    "data_cleaning",
    "indepth_eda",
    "feature_engineering",
    "model_building",
)


def _next_execution_phase(phase: str) -> str:
    """Advance within execution phases only (never to summary phases)."""
    try:
        idx = EXECUTION_PHASES.index(phase)
    except ValueError:
        return EXECUTION_PHASES[0]
    if idx + 1 < len(EXECUTION_PHASES):
        return EXECUTION_PHASES[idx + 1]
    return phase


async def planner_node(state: GraphState) -> dict:
    
    status = state.review_state.latest_review_status
    phase =  state.current_phase

    # Default to whatever phase we're already in, if it's a supported execution phase.
    target_phase = (state.current_phase
        if state.current_phase in EXECUTION_PHASES
        else EXECUTION_PHASES[0]
    )

    if status == "retry" and phase in EXECUTION_PHASES:
        target_phase = phase
    elif status == "approved" and phase != "model_building":
        target_phase = _next_execution_phase(phase)
        
    if target_phase not in PHASE_SEQUENCE:
        raise ValueError(
            f"Invalid phase computed in planner_node: {target_phase}. "
            f"Expected phase set: {PHASE_SEQUENCE}"
        )
    print(f"-------------------------------- {target_phase} --------------------------------")
    # Ensure the planner agent sees the correct phase in its prompts/allowlists.
    state.current_phase = target_phase

    updates = await timed_node(state, node_name="planner", runner=run_planner)
    return {
        **updates,
        "current_phase": target_phase,
        "phase_history": [target_phase],
    }
