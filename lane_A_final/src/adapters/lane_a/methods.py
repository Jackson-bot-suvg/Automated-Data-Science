"""Gate policies for the method dictionary (F0, M0, M2, M3, M4, M5, G).

All variants reuse the *same* deterministic F0 check functions
(``app.tools.reviewer_tools.review_*`` and ``BackgroundReviewerAgent._check_reader_output``).
They differ only in which computed failures are *enforced*, and in the retry
semantics.  Every computed verdict is logged (``gate_verdict``) whether or not it
is enforced, so shadow judgements are recoverable offline without being shown to
the agent.

Failure classes
  S  semantic task-contract failures (target / submission contract)  -> regex r"target|submission"
  E  execution / structural failures (missing artifact, unreadable CSV, empty model list, ...)
"""

from __future__ import annotations

import re
from typing import Any

from lane_a.context import ctx
from lane_a.state_views import candidate_hash, contract_digest, state_hash

SEMANTIC_RE = re.compile(r"target|submission", re.IGNORECASE)
PATH_TOKEN_RE = re.compile(r"\S*[/\\]\S*")  # artifact paths such as .../predictions/submission.csv

METHODS = (
    "F0_ORIGINAL",
    "M0_NO_SEMANTIC_ENFORCEMENT",
    "M0_NAIVE_GATE_REMOVAL",
    "M2_INTAKE_ONLY_STRONG",
    "M3_SHADOW",
    "M4_END_ONLY",
    "M5_BLOCK_ONLY",
    "G_GENERIC_RETRY",
)

PHASE_ORDER = (
    "reader",
    "preliminary_eda",
    "data_cleaning",
    "indepth_eda",
    "feature_engineering",
    "model_building",
)

MAX_ATTEMPTS = 3  # F0: ReviewerAgent.max_retries_per_phase; fatal when attempt >= 3 (=> at most 2 retries)


def failure_class(text: str) -> str:
    """S = target/submission contract failure; E = execution/structural failure.

    Paths are stripped first so that e.g. a missing ``.../submission.csv`` artifact is E.
    """

    return "S" if SEMANTIC_RE.search(PATH_TOKEN_RE.sub(" ", text or "")) else "E"


def _enforced(failures: list[str], *, gate: str, phase: str, method: str) -> list[str]:
    """Return the subset of computed blocking failures that the method enforces."""

    if method in {"F0_ORIGINAL", "G_GENERIC_RETRY", "M5_BLOCK_ONLY"}:
        return list(failures)
    if method == "M3_SHADOW":
        return []
    if method == "M0_NO_SEMANTIC_ENFORCEMENT":
        return [f for f in failures if failure_class(f) == "E"]
    if method == "M2_INTAKE_ONLY_STRONG":
        if gate == "background_reviewer":
            return list(failures)
        return [f for f in failures if failure_class(f) == "E"]
    if method == "M4_END_ONLY":
        # intermediate gates are shadowed; the end check is handled separately at model_building
        if gate == "reviewer" and phase == "model_building":
            return list(failures)
        return []
    if method == "M0_NAIVE_GATE_REMOVAL":
        return []  # reviewer nodes are not even in the graph for this variant
    raise ValueError(f"unknown method {method}")


def _f0_status(n_failures: int, attempt: int, hard_fatal: bool = False) -> str:
    if n_failures == 0:
        return "approved"
    if hard_fatal or attempt >= MAX_ATTEMPTS:
        return "fatal"
    return "retry"


def _next_action(gate: str, phase: str, status: str) -> str:
    if status == "fatal":
        return "END(aborted)"
    if gate == "background_reviewer":
        return "planner" if status == "approved" else "reader(retry)"
    if status == "approved":
        return "summarizer" if phase == "model_building" else "planner(next_phase)"
    return f"planner(retry:{phase})"


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def install_method(method: str) -> None:
    from app.agents import bg_reviewer as bg_mod
    from app.agents import reviewer_agent as rv_mod
    from app.agents import planner as planner_mod
    from app.schemas.reviewer_schema import ReviewCheckResult
    from app.tools import reviewer_tools

    if method not in METHODS:
        raise ValueError(f"unknown method {method}")

    ReviewerAgent = rv_mod.ReviewerAgent
    BackgroundReviewerAgent = bg_mod.BackgroundReviewerAgent

    orig_phase_review = ReviewerAgent._run_phase_review
    orig_phase_status = ReviewerAgent._status_for
    orig_phase_run = ReviewerAgent.run
    orig_bg_check = BackgroundReviewerAgent._check_reader_output
    orig_bg_status = BackgroundReviewerAgent._status_for
    orig_retry_ctx = planner_mod.PlannerAgent._retry_context_for_phase

    # ------------------------------------------------------------------ phase reviewer
    def patched_phase_review(self, phase: str, state):
        c = ctx()
        computed = orig_phase_review(self, phase, state)
        attempt = int(state.review_state.retry_count_by_phase.get(phase, 0)) + 1
        failures = list(computed.blocking_failures)

        if method == "M4_END_ONLY" and phase == "model_building":
            failures = _m4_end_check_failures(state, reviewer_tools, bg_mod, orig_bg_check)
        enforced = _enforced(failures, gate="reviewer", phase=phase, method=method)
        f0_status = _f0_status(len(computed.blocking_failures), attempt)
        c.gate_counter[f"reviewer:{phase}"] = c.gate_counter.get(f"reviewer:{phase}", 0) + 1
        c._pending_gate = {  # type: ignore[attr-defined]
            "gate": "reviewer",
            "phase": phase,
            "attempt": attempt,
            "computed_failures": failures,
            "enforced_failures": enforced,
            "f0_status": f0_status,
        }
        c.logger.emit(
            "gate_check",
            phase=phase,
            actor="reviewer",
            state_hash=state_hash(state, c.path_replacements),
            candidate_hash=candidate_hash(state, phase, c.path_replacements),
            contract_digest=contract_digest(state),
            source_fields=list(computed.checked_state_fields),
            verdict=f0_status,
            payload={
                "method": method,
                "attempt": attempt,
                "computed_failures": [{"text": f, "class": failure_class(f)} for f in failures],
                "enforced_failures": enforced,
                "warnings": list(computed.warnings),
                "checked_artifacts": list(computed.checked_artifacts),
                "occurrence": c.gate_counter[f"reviewer:{phase}"],
                "details": computed.details,
            },
        )
        return ReviewCheckResult(
            phase=computed.phase,
            blocking_failures=enforced,
            warnings=list(computed.warnings),
            checked_artifacts=list(computed.checked_artifacts),
            checked_state_fields=list(computed.checked_state_fields),
            details=computed.details,
        )

    def patched_phase_status(self, check_result, attempt):
        c = ctx()
        pending = getattr(c, "_pending_gate", {}) or {}
        phase = pending.get("phase", check_result.phase)
        if method == "M5_BLOCK_ONLY" and not check_result.passed:
            status = "fatal"
        elif method == "M4_END_ONLY" and phase == "model_building" and not check_result.passed:
            if c.end_check_intake_failed:
                status = "fatal"
            else:
                status = orig_phase_status(self, check_result, attempt)
        else:
            status = orig_phase_status(self, check_result, attempt)
        c.logger.emit(
            "gate_verdict",
            phase=phase,
            actor="reviewer",
            verdict=status,
            next_action=_next_action("reviewer", phase, status),
            payload={
                "method": method,
                "attempt": attempt,
                "enforced": bool(pending.get("enforced_failures")),
                "f0_counterfactual_status": pending.get("f0_status"),
                "computed_failure_count": len(pending.get("computed_failures") or []),
                "enforced_failure_count": len(pending.get("enforced_failures") or []),
                "shadow": method in {"M3_SHADOW", "M4_END_ONLY", "M0_NO_SEMANTIC_ENFORCEMENT", "M2_INTAKE_ONLY_STRONG"}
                and len(pending.get("computed_failures") or []) != len(pending.get("enforced_failures") or []),
            },
        )
        c._last_phase_status = status  # type: ignore[attr-defined]
        return status

    async def patched_phase_run(self, state):
        c = ctx()
        c.end_check_intake_failed = False
        c.m4_redirect_phase = ""
        update = await orig_phase_run(self, state)
        phase = state.current_phase or state.review_state.pending_review_phase
        status = (update.get("review_state") or {}).get("latest_review_status")
        if method == "M4_END_ONLY" and phase == "model_building" and status == "retry" and c.m4_redirect_phase:
            update["current_phase"] = c.m4_redirect_phase
            update["review_state"]["pending_review_phase"] = c.m4_redirect_phase
            c.logger.emit(
                "m4_end_remediation",
                phase=phase,
                actor="reviewer",
                next_action=f"planner(retry:{c.m4_redirect_phase})",
                payload={"redirect_phase": c.m4_redirect_phase},
            )
        return update

    # ------------------------------------------------------------------ intake reviewer
    def patched_bg_check(self, state):
        c = ctx()
        computed, hard_fatal = orig_bg_check(self, state)
        attempt = int(state.review_state.retry_count_by_phase.get("reader", 0)) + 1
        failures = list(computed.blocking_failures)
        enforced = _enforced(failures, gate="background_reviewer", phase="reader", method=method)
        enforced_hard = bool(hard_fatal and any(failure_class(f) == "E" for f in enforced))
        f0_status = _f0_status(len(failures), attempt, hard_fatal)
        c.gate_counter["background_reviewer:reader"] = c.gate_counter.get("background_reviewer:reader", 0) + 1
        c._pending_gate = {  # type: ignore[attr-defined]
            "gate": "background_reviewer",
            "phase": "reader",
            "attempt": attempt,
            "computed_failures": failures,
            "enforced_failures": enforced,
            "f0_status": f0_status,
        }
        c.logger.emit(
            "gate_check",
            phase="reader",
            actor="background_reviewer",
            state_hash=state_hash(state, c.path_replacements),
            candidate_hash=candidate_hash(state, "reader", c.path_replacements),
            contract_digest=contract_digest(state),
            source_fields=list(computed.checked_state_fields),
            verdict=f0_status,
            payload={
                "method": method,
                "attempt": attempt,
                "hard_fatal": bool(hard_fatal),
                "computed_failures": [{"text": f, "class": failure_class(f)} for f in failures],
                "enforced_failures": enforced,
                "warnings": list(computed.warnings),
                "occurrence": c.gate_counter["background_reviewer:reader"],
            },
        )
        sanitized = ReviewCheckResult(
            phase=computed.phase,
            blocking_failures=enforced,
            warnings=list(computed.warnings),
            checked_artifacts=list(computed.checked_artifacts),
            checked_state_fields=list(computed.checked_state_fields),
            details=computed.details,
        )
        return sanitized, enforced_hard

    def patched_bg_status(self, check_result, attempt, hard_fatal):
        c = ctx()
        if method == "M5_BLOCK_ONLY" and not check_result.passed:
            status = "fatal"
        else:
            status = orig_bg_status(self, check_result, attempt, hard_fatal)
        pending = getattr(c, "_pending_gate", {}) or {}
        c.logger.emit(
            "gate_verdict",
            phase="reader",
            actor="background_reviewer",
            verdict=status,
            next_action=_next_action("background_reviewer", "reader", status),
            payload={
                "method": method,
                "attempt": attempt,
                "f0_counterfactual_status": pending.get("f0_status"),
                "computed_failure_count": len(pending.get("computed_failures") or []),
                "enforced_failure_count": len(pending.get("enforced_failures") or []),
            },
        )
        return status

    # ------------------------------------------------------------------ generic retry (G)
    def patched_retry_ctx(self, *, phase, profile_summary):
        result = orig_retry_ctx(self, phase=phase, profile_summary=profile_summary)
        if result is None:
            return None
        withheld = {
            "blocking_failures": len(result.get("blocking_failures") or []),
            "warnings": len(result.get("warnings") or []),
        }
        result = dict(result)
        result["blocking_failures"] = []
        result["warnings"] = []
        ctx().logger.emit(
            "retry_context",
            phase=phase,
            actor="planner",
            payload={"method": method, "diagnostics_withheld": withheld},
        )
        return result

    ReviewerAgent._run_phase_review = patched_phase_review
    ReviewerAgent._status_for = patched_phase_status
    ReviewerAgent.run = patched_phase_run
    BackgroundReviewerAgent._check_reader_output = patched_bg_check
    BackgroundReviewerAgent._status_for = patched_bg_status
    if method == "G_GENERIC_RETRY":
        planner_mod.PlannerAgent._retry_context_for_phase = patched_retry_ctx


def _m4_end_check_failures(state, reviewer_tools, bg_mod, orig_bg_check) -> list[str]:
    """M4 end check: the same deterministic F0 rules, applied once to the final state."""

    c = ctx()
    per_phase: dict[str, list[str]] = {}
    intake_result, _hard = orig_bg_check(bg_mod.BackgroundReviewerAgent(), state)
    per_phase["reader"] = list(intake_result.blocking_failures)
    funcs = {
        "preliminary_eda": reviewer_tools.review_preliminary_eda,
        "data_cleaning": reviewer_tools.review_data_cleaning,
        "indepth_eda": reviewer_tools.review_indepth_eda,
        "feature_engineering": reviewer_tools.review_feature_engineering,
        "model_building": reviewer_tools.review_model_building,
    }
    for phase, fn in funcs.items():
        try:
            per_phase[phase] = list(fn(state).blocking_failures)
        except Exception as exc:  # noqa: BLE001
            per_phase[phase] = [f"end-check error: {exc!r}"]
    earliest = next((p for p in PHASE_ORDER if per_phase.get(p)), "")
    c.end_check_intake_failed = earliest == "reader"
    c.m4_redirect_phase = earliest if earliest and earliest not in {"reader", "model_building"} else ""
    c.logger.emit(
        "m4_end_check",
        phase="model_building",
        actor="reviewer",
        payload={"per_phase_failures": per_phase, "earliest_failing_phase": earliest},
    )
    out = []
    for phase in PHASE_ORDER:
        out.extend(f"[{phase}] {f}" for f in per_phase.get(phase, []))
    return out
