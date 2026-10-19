"""Main runner: build the case matrix, run M1/M2/F0, and persist judgments + results.

Usage:
    python -m lane_B.runner            # run everything and write CSV outputs
    python -m lane_B.runner --dry-run  # print the plan matrix without running
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from lane_B.cases import build_cases, write_manifest
from lane_B.checkers import METHODS, load_package
from lane_B.mutators import apply_mutation, case_data_dir
from lane_B.oracle import judge
from lane_B.registry import OUTPUT_ROOT, build_registry


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _print_summary(results: list[dict]) -> None:
    print("\nPer-method outcome summary (TP/FN/FP/TN):")
    methods = ["M1", "M2", "F0"]
    for method_id in methods:
        counts = Counter(
            row["outcome"] for row in results if row["method_id"] == method_id
        )
        line = " | ".join(
            f"{key}={counts.get(key, 0)}" for key in ("TP", "FN", "FP", "TN")
        )
        print(f"  {method_id}: {line}")


def run(*, dry_run: bool = False) -> tuple[Path, Path]:
    contracts = build_registry()
    cases = build_cases(contracts)
    manifest_path = write_manifest(cases)

    if dry_run:
        print(f"Planned matrix: {len(cases)} cases x {len(METHODS)} methods")
        for case in cases:
            print(
                f"  {case.case_id}  [{case.truth_status}] "
                f"expected={case.expected_safe_action}"
            )
        return manifest_path, OUTPUT_ROOT / "results.csv"

    # (Re)generate the mutated visible packages from pristine source data.
    for case in cases:
        apply_mutation(case, contracts[case.dataset_key])

    judgment_rows: list[dict] = []
    result_rows: list[dict] = []

    for case in cases:
        package = load_package(case_data_dir(case.case_id))
        for method_id, checker in METHODS.items():
            judgment = checker(package)
            judgment_rows.append(
                {
                    "case_id": case.case_id,
                    "dataset_key": case.dataset_key,
                    "template_id": case.template_id,
                    "method_id": method_id,
                    "verdict": judgment.verdict,
                    "root_causes": ",".join(judgment.root_causes),
                    "earliest_stage": judgment.earliest_stage,
                    "reasons": "; ".join(judgment.details.get("reasons", [])),
                }
            )
            result = judge(case, method_id, judgment)
            result_rows.append(
                {
                    "case_id": case.case_id,
                    "dataset_key": case.dataset_key,
                    "template_id": case.template_id,
                    "method_id": method_id,
                    "truth_status": case.truth_status,
                    "expected_safe_action": case.expected_safe_action,
                    "verdict": judgment.verdict,
                    "outcome": result.outcome,
                    "localization": result.localization,
                }
            )

    judgments_path = _write_csv(
        OUTPUT_ROOT / "judgments.csv",
        [
            "case_id",
            "dataset_key",
            "template_id",
            "method_id",
            "verdict",
            "root_causes",
            "earliest_stage",
            "reasons",
        ],
        judgment_rows,
    )
    results_path = _write_csv(
        OUTPUT_ROOT / "results.csv",
        [
            "case_id",
            "dataset_key",
            "template_id",
            "method_id",
            "truth_status",
            "expected_safe_action",
            "verdict",
            "outcome",
            "localization",
        ],
        result_rows,
    )

    print(f"case manifest: {manifest_path}")
    print(f"judgments:     {judgments_path}  ({len(judgment_rows)} rows)")
    print(f"results:       {results_path}  ({len(result_rows)} rows)")
    _print_summary(result_rows)
    return judgments_path, results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lane_B fault-injection harness.")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix without running")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
