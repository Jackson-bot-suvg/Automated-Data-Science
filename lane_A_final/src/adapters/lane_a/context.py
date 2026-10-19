"""Process-wide run context shared by observers, method policies and injections."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lane_a.common import EventLogger


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    hooks_dir: Path
    logger: EventLogger
    method_id: str = "F0_ORIGINAL"
    case_id: str = ""
    template_id: str = "C00"
    injection: dict[str, Any] = field(default_factory=dict)
    fork: dict[str, Any] = field(default_factory=dict)
    seed_block: int = 0
    path_replacements: dict[str, str] = field(default_factory=dict)
    # mutable runtime bookkeeping
    current_phase: str = ""
    gate_counter: dict[str, int] = field(default_factory=dict)
    injections_applied: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    llm_errors: int = 0
    llm_budget: int = 60
    fit_seconds_total: float = 0.0
    last_predict: dict[str, Any] = field(default_factory=dict)
    end_check_intake_failed: bool = False
    m4_redirect_phase: str = ""
    shadow_records: list[dict[str, Any]] = field(default_factory=list)


_CTX: RunContext | None = None


def set_ctx(ctx: RunContext) -> None:
    global _CTX
    _CTX = ctx


def ctx() -> RunContext:
    if _CTX is None:
        raise RuntimeError("lane_a run context is not initialised")
    return _CTX


def has_ctx() -> bool:
    return _CTX is not None


class LLMBudgetExceeded(BaseException):
    """Frozen per-run LLM call budget exhausted (S8/budget).

    BaseException on purpose: ``app.llm.helpers`` swallows ``Exception`` and falls
    back silently, which would hide a budget stop.
    """


class ForkPointReached(BaseException):
    """Raised (as BaseException so agent code cannot swallow it) at the fork point of a prefix run."""
