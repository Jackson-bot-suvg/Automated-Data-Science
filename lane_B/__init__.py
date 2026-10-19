"""lane_B: deterministic semantic-fault injection harness (RTS-TAI-PARALLEL-v4.0, role B).

This package reproduces the core "basic validation" experiment: manufacture
ground-truth target/output-column semantic faults on DEV datasets, then compare
three judgment modes (M1 structural-only, M2 intake-only, F0 full gating).
"""

from __future__ import annotations

__all__ = ["registry", "cases", "mutators"]
