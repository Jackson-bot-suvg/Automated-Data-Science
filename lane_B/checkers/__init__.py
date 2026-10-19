"""Three deterministic judgment methods for the lane_B harness.

* M1 — structural-only checks (no semantic intent)
* M2 — intake-only contract resolution + entry semantic checks
* F0 — full gating (M2 + post-intake continuous checks)
"""

from __future__ import annotations

from lane_B.checkers.base import Judgment, VisiblePackage, load_package
from lane_B.checkers.f0_full import check_f0
from lane_B.checkers.m1_structural import check_m1
from lane_B.checkers.m2_intake import check_m2

METHODS = {"M1": check_m1, "M2": check_m2, "F0": check_f0}

__all__ = [
    "Judgment",
    "METHODS",
    "VisiblePackage",
    "check_f0",
    "check_m1",
    "check_m2",
    "load_package",
]
