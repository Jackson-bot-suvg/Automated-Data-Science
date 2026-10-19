"""F0_ORIGINAL: full contract gating (intake + post-intake continuous checks).

Runs the M2 intake checks first; if they pass, it additionally re-validates the
contract against the downstream artifacts (prediction output / row mapping), so
post-intake drift (e.g. F05) is caught where intake-only and structural-only
methods cannot see it.
"""

from __future__ import annotations

from lane_B.checkers.base import (
    Judgment,
    VisiblePackage,
    intake_semantic_checks,
    post_intake_checks,
    resolve_intake,
)


def check_f0(package: VisiblePackage) -> Judgment:
    resolution = resolve_intake(package)

    intake_causes = intake_semantic_checks(resolution, package)
    if intake_causes:
        fields = list({field for field, _ in intake_causes})
        return Judgment(
            method_id="F0",
            verdict="block",
            root_causes=fields,
            earliest_stage="intake",
            details={"reasons": [reason for _, reason in intake_causes], "source": resolution.source},
        )

    post_causes = post_intake_checks(resolution, package)
    if post_causes:
        fields = list({field for field, _ in post_causes})
        return Judgment(
            method_id="F0",
            verdict="block",
            root_causes=fields,
            earliest_stage="post_intake",
            details={"reasons": [reason for _, reason in post_causes]},
        )

    return Judgment(method_id="F0", verdict="pass", earliest_stage="none")


__all__ = ["check_f0"]
