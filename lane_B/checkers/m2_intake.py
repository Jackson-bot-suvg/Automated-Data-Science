"""M2_INTAKE_ONLY_STRONG: resolve the contract at intake and check once.

Reuses ReaderAgent's deterministic contract resolution, then applies the entry
semantic checks. It performs NO post-intake / later-stage re-validation, so
drift that only appears after intake (e.g. F05) is invisible to this method.
"""

from __future__ import annotations

from lane_B.checkers.base import (
    Judgment,
    VisiblePackage,
    intake_semantic_checks,
    resolve_intake,
)


def check_m2(package: VisiblePackage) -> Judgment:
    resolution = resolve_intake(package)
    causes = intake_semantic_checks(resolution, package)

    if not causes:
        return Judgment(method_id="M2", verdict="pass", earliest_stage="none")

    fields = list({field for field, _ in causes})
    return Judgment(
        method_id="M2",
        verdict="block",
        root_causes=fields,
        earliest_stage="intake",
        details={"reasons": [reason for _, reason in causes], "source": resolution.source},
    )


__all__ = ["check_m2"]
