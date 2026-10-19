"""Case specifications and template definitions for the lane_B harness.

A *case* is one (dataset, template) pair. Each template declares whether it is a
clean control, a legal variation (must NOT be blocked), or a semantic fault
(must be blocked), together with the fault fields the oracle expects to be hit.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from lane_B.registry import GoldContract, OUTPUT_ROOT

# Safe-action expectations consumed by the offline oracle.
PASS = "pass"  # correct system continues (no block / clarify)
BLOCK = "block"  # correct system blocks or asks to clarify


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    dataset_key: str
    template_id: str
    condition: str
    truth_status: str  # "clean" | "legal" | "fault"
    target_fault_fields: tuple[str, ...]
    expected_safe_action: str  # PASS | BLOCK
    injection_stage: str  # "intake" | "post_intake"


# Template registry. The mutator for each template lives in mutators.py.
# truth_status: clean (control) / legal (variation) / fault (injected defect).
TEMPLATES: dict[str, dict] = {
    "C00": {
        "truth_status": "clean",
        "expected_safe_action": PASS,
        "target_fault_fields": (),
        "injection_stage": "intake",
        "condition": "original clean task (no mutation)",
    },
    "B02": {
        "truth_status": "legal",
        "expected_safe_action": PASS,
        "target_fault_fields": (),
        "injection_stage": "intake",
        "condition": "legal variation: sample-submission column order changed",
    },
    "F01": {
        "truth_status": "fault",
        "expected_safe_action": BLOCK,
        "target_fault_fields": ("target_columns",),
        "injection_stage": "intake",
        "condition": "fault: target column(s) removed from train.csv",
    },
    "F02": {
        "truth_status": "fault",
        "expected_safe_action": BLOCK,
        "target_fault_fields": ("target_columns",),
        "injection_stage": "intake",
        "condition": "fault: target column(s) leaked into test.csv",
    },
    "F03": {
        "truth_status": "fault",
        "expected_safe_action": BLOCK,
        "target_fault_fields": ("target_columns",),
        "injection_stage": "intake",
        "condition": "fault: task text points to the wrong target column",
    },
    "F04": {
        "truth_status": "fault",
        "expected_safe_action": BLOCK,
        "target_fault_fields": ("output_columns",),
        "injection_stage": "intake",
        "condition": "fault: sample-submission output column renamed to a wrong name",
    },
    "F05": {
        "truth_status": "fault",
        "expected_safe_action": BLOCK,
        "target_fault_fields": ("target_columns", "output_columns"),
        "injection_stage": "post_intake",
        "condition": "fault: post-intake drift (prediction output columns no longer match the contract)",
    },
}


def build_cases(contracts: dict[str, GoldContract]) -> list[CaseSpec]:
    """Instantiate the full case list: every applicable template per dataset."""

    cases: list[CaseSpec] = []
    for dataset_key, contract in contracts.items():
        for template_id, meta in TEMPLATES.items():
            cases.append(
                CaseSpec(
                    case_id=f"{dataset_key}__{template_id}",
                    dataset_key=dataset_key,
                    template_id=template_id,
                    condition=str(meta["condition"]),
                    truth_status=str(meta["truth_status"]),
                    target_fault_fields=tuple(meta["target_fault_fields"]),
                    expected_safe_action=str(meta["expected_safe_action"]),
                    injection_stage=str(meta["injection_stage"]),
                )
            )
    return cases


def write_manifest(cases: list[CaseSpec], output_dir: Path = OUTPUT_ROOT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "case_manifest.csv"
    fieldnames = [
        "case_id",
        "dataset_key",
        "template_id",
        "condition",
        "truth_status",
        "target_fault_fields",
        "expected_safe_action",
        "injection_stage",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "dataset_key": case.dataset_key,
                    "template_id": case.template_id,
                    "condition": case.condition,
                    "truth_status": case.truth_status,
                    "target_fault_fields": ",".join(case.target_fault_fields),
                    "expected_safe_action": case.expected_safe_action,
                    "injection_stage": case.injection_stage,
                }
            )
    return path


__all__ = ["BLOCK", "PASS", "CaseSpec", "TEMPLATES", "build_cases", "write_manifest"]
