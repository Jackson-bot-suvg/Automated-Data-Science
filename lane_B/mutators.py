"""File-level semantic-fault and legal-variation mutators.

Every mutator works on a COPY of the original dataset, written under
``outputs/cases/<case_id>/data/``. The original ``data/`` directory is never
modified, keeping runs traceable and the ground truth isolated from the
agent-visible package.

Each case directory contains (after mutation):

* ``train.csv`` / ``test.csv`` / ``sample_submission.csv`` — possibly mutated
* ``overview.txt`` — copied from the source dataset (kept for context)
* ``task.txt`` — the visible natural-language task (mutated for F03)
* ``prediction.csv`` — the "model-building output" the F0 post-intake check
  inspects; correct for every case except F05 (where it drifts)
* ``case.json`` — reproducible mutation metadata (no ground-truth target names)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd

from lane_B.cases import CaseSpec
from lane_B.registry import GoldContract, OUTPUT_ROOT


def case_data_dir(case_id: str) -> Path:
    return OUTPUT_ROOT / "cases" / case_id / "data"


def _copy_source_files(contract: GoldContract, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "test.csv", "sample_submission.csv", "overview.txt"):
        source = contract.dataset_root / name
        if source.exists():
            shutil.copy2(source, data_dir / name)


def _visible_task(case: CaseSpec, contract: GoldContract) -> str:
    """The natural-language task visible to the agent (mutated for F03)."""

    if case.template_id == "F03":
        wrong = _wrong_target_column(contract)
        return f"predict the value of column '{wrong}'"
    return contract.task


def _wrong_target_column(contract: GoldContract) -> str:
    """Pick a deterministic non-id, non-target feature column as the 'wrong' target."""

    train_path = contract.dataset_root / "train.csv"
    columns = pd.read_csv(train_path, nrows=0).columns.tolist()
    excluded = {c.casefold() for c in contract.id_columns} | {
        c.casefold() for c in contract.target_columns
    }
    for column in columns:
        if column.casefold() not in excluded:
            return str(column)
    return str(columns[0])


def _write_prediction(contract: GoldContract, data_dir: Path, *, drifted: bool) -> None:
    """Emit the post-intake prediction artifact that F0's model check inspects."""

    test_path = data_dir / "test.csv"
    id_column = contract.id_columns[0] if contract.id_columns else None
    if id_column is None or not test_path.exists():
        return
    try:
        test_columns = pd.read_csv(test_path, nrows=0).columns.tolist()
        if id_column not in test_columns:
            return
        ids = pd.read_csv(test_path, usecols=[id_column])
    except Exception:
        return

    if drifted:
        output_columns = [f"Drifted_{column}" for column in contract.output_columns]
    else:
        output_columns = list(contract.output_columns)

    fill_value: float | int = 0.0 if contract.task_family == "regression" else 0
    prediction = pd.DataFrame({id_column: ids[id_column]})
    for column in output_columns:
        prediction[column] = fill_value
    prediction.to_csv(data_dir / "prediction.csv", index=False)


def _mut_reorder(contract: GoldContract, data_dir: Path) -> list[str]:
    submission_path = data_dir / "sample_submission.csv"
    frame = pd.read_csv(submission_path)
    frame = frame[list(reversed(frame.columns))]
    frame.to_csv(submission_path, index=False)
    return ["sample_submission.csv"]


def _mut_target_missing(contract: GoldContract, data_dir: Path) -> list[str]:
    train_path = data_dir / "train.csv"
    frame = pd.read_csv(train_path)
    dropped = [c for c in contract.target_columns if c in frame.columns]
    if dropped:
        frame = frame.drop(columns=dropped)
        frame.to_csv(train_path, index=False)
    return ["train.csv"]


def _mut_target_leak(contract: GoldContract, data_dir: Path) -> list[str]:
    test_path = data_dir / "test.csv"
    frame = pd.read_csv(test_path)
    fill_value: float | int = 0.0 if contract.task_family == "regression" else 0
    for column in contract.target_columns:
        if column not in frame.columns:
            frame[column] = fill_value
    frame.to_csv(test_path, index=False)
    return ["test.csv"]


def _mut_output_mismatch(contract: GoldContract, data_dir: Path) -> list[str]:
    submission_path = data_dir / "sample_submission.csv"
    frame = pd.read_csv(submission_path)
    rename = {
        column: f"Wrong_{column}"
        for column in contract.output_columns
        if column in frame.columns
    }
    if rename:
        frame = frame.rename(columns=rename)
        frame.to_csv(submission_path, index=False)
    return ["sample_submission.csv"]


_MUTATORS: dict[str, Callable[[GoldContract, Path], list[str]]] = {
    "B02": _mut_reorder,
    "F01": _mut_target_missing,
    "F02": _mut_target_leak,
    "F04": _mut_output_mismatch,
}


def apply_mutation(case: CaseSpec, contract: GoldContract) -> Path:
    """Produce the mutated visible package for a case and return its data dir."""

    data_dir = case_data_dir(case.case_id)
    _copy_source_files(contract, data_dir)

    task_text = _visible_task(case, contract)
    (data_dir / "task.txt").write_text(task_text, encoding="utf-8")

    changed: list[str] = []
    mutator = _MUTATORS.get(case.template_id)
    if mutator is not None:
        changed = mutator(contract, data_dir)

    # F05 (post-intake drift) keeps the intake package clean and only corrupts
    # the downstream prediction artifact.
    _write_prediction(contract, data_dir, drifted=(case.template_id == "F05"))
    changed.append("prediction.csv")

    _write_case_metadata(case, contract, data_dir, changed)
    return data_dir


def _write_case_metadata(
    case: CaseSpec,
    contract: GoldContract,
    data_dir: Path,
    changed: list[str],
) -> None:
    payload = {
        "case_id": case.case_id,
        "dataset_key": case.dataset_key,
        "template_id": case.template_id,
        "condition": case.condition,
        "truth_status": case.truth_status,
        "expected_safe_action": case.expected_safe_action,
        "injection_stage": case.injection_stage,
        "task_family": contract.task_family,
        "visible_task": _visible_task(case, contract),
        "changed_files": changed,
    }
    (data_dir / "case.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


__all__ = ["apply_mutation", "case_data_dir"]
