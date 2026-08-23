from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.schemas.state_schema import GraphState


async def timed_node(
    state: GraphState,
    *,
    node_name: str,
    runner: Callable[[GraphState], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    start = time.perf_counter()
    updates = await runner(state)
    elapsed = round(time.perf_counter() - start, 4)

    runtime = dict(state.runtime_state.model_dump(mode="python"))
    existing = dict(runtime.get("phase_runtime_sec") or {})
    existing[node_name] = float(existing.get(node_name, 0.0) or 0.0) + elapsed
    runtime["phase_runtime_sec"] = existing
    runtime["total_runtime_sec"] = round(
        float(runtime.get("total_runtime_sec", 0.0) or 0.0) + elapsed,
        4,
    )

    merged_runtime = dict(updates.get("runtime_state") or {})
    merged_runtime = {**runtime, **merged_runtime}
    updates["runtime_state"] = merged_runtime
    return updates
