"""M1_STRUCTURAL_STRONG: pure structural checks, no semantic intent.

Only validates file readability, non-emptiness and column-name integrity. It
never reads the task text or the target/output semantics, so structural-only
checks cannot flag the semantic faults injected by this harness.
"""

from __future__ import annotations

from lane_B.checkers.base import Judgment, VisiblePackage


def check_m1(package: VisiblePackage) -> Judgment:
    causes: list[tuple[str, str]] = []

    if not package.train_columns:
        causes.append(("structure", "train.csv is missing, empty, or unreadable"))
    if not package.test_columns:
        causes.append(("structure", "test.csv is missing, empty, or unreadable"))
    if not package.submission_columns:
        causes.append(("structure", "sample_submission.csv is missing, empty, or unreadable"))

    for label, columns in (
        ("train", package.train_columns),
        ("test", package.test_columns),
        ("submission", package.submission_columns),
        ("prediction", package.prediction_columns),
    ):
        if len(columns) != len(set(columns)):
            causes.append(("structure", f"{label} contains duplicate column names"))

    if package.has_prediction and not package.prediction_columns:
        causes.append(("structure", "prediction.csv is empty or unreadable"))

    if not causes:
        return Judgment(method_id="M1", verdict="pass", earliest_stage="none")

    fields = sorted({field for field, _ in causes})
    return Judgment(
        method_id="M1",
        verdict="block",
        root_causes=fields,
        earliest_stage="intake",
        details={"reasons": [reason for _, reason in causes]},
    )


__all__ = ["check_m1"]
