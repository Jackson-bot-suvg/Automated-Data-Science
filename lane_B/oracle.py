"""Offline oracle: classify each (case, method) judgment against ground truth.

Ground truth is the CaseSpec (truth_status + target_fault_fields + expected
safe action), NOT the hidden C_gold values. Outcomes:

* fault  + block -> TP (detected)
* fault  + pass  -> FN (missed)
* legal/clean + pass -> TN (correctly not blocked)
* legal/clean + block -> FP (false block)

Root-cause localization is reported for detected faults only: exact (all fault
fields hit), partial (some hit), none (no fault field hit).
"""

from __future__ import annotations

from dataclasses import dataclass

from lane_B.cases import CaseSpec
from lane_B.checkers.base import Judgment


@dataclass(frozen=True)
class OracleResult:
    case_id: str
    method_id: str
    outcome: str  # "TP" | "FN" | "FP" | "TN"
    localization: str  # "exact" | "partial" | "none" | "n_a"


def judge(case: CaseSpec, method_id: str, judgment: Judgment) -> OracleResult:
    is_fault = case.truth_status == "fault"
    detected = judgment.verdict == "block"

    if is_fault:
        outcome = "TP" if detected else "FN"
    else:
        outcome = "FP" if detected else "TN"

    return OracleResult(
        case_id=case.case_id,
        method_id=method_id,
        outcome=outcome,
        localization=_localization(case, judgment, outcome),
    )


def _localization(case: CaseSpec, judgment: Judgment, outcome: str) -> str:
    if outcome != "TP":
        return "none" if case.truth_status == "fault" else "n_a"

    true_fields = set(case.target_fault_fields)
    detected_fields = set(judgment.root_causes)
    if not true_fields:
        return "n_a"
    if detected_fields == true_fields:
        return "exact"
    if detected_fields & true_fields:
        return "partial"
    return "none"


__all__ = ["OracleResult", "judge"]
