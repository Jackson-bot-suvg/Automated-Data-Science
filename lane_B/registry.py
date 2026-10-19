"""Ground-truth contract registry (C_gold) for the lane_B fault-injection harness.

``C_gold`` is derived ONLY from two sources:

* ``app.validation_matrix`` (the project's built-in dataset specs), and
* each dataset's ``sample_submission.csv`` (the authoritative submission schema).

It must stay isolated from the "visible materials" handed to the checkers,
mirroring the ground-truth / agent-visible isolation required by
RTS-TAI-PARALLEL-v4.0 (section 3: private ``C_gold`` vs ``C_read``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.utils.target_scope import normalize_target_columns
from app.validation_matrix import ALL_VALIDATION_DATASETS, DatasetValidationSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# Four representative DEV datasets covering all four canonical task families.
EVAL_DATASET_KEYS: tuple[str, ...] = (
    "spaceship_titanic",  # binary_classification
    "academic_success",  # multiclass_classification
    "house_prices",  # regression
    "plate_defect",  # multilabel_classification
)


@dataclass(frozen=True)
class GoldContract:
    """The authoritative task contract for one dataset (oracle ground truth)."""

    dataset_key: str
    task: str
    target_columns: tuple[str, ...]
    task_family: str
    target_mode: str
    metric_hint: str
    dataset_root: Path
    id_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    submission_columns: tuple[str, ...]

    @property
    def submission_path(self) -> Path:
        return self.dataset_root / "sample_submission.csv"


def _specs() -> dict[str, DatasetValidationSpec]:
    return {spec.key: spec for spec in ALL_VALIDATION_DATASETS}


def _resolve_submission_columns(dataset_root: Path) -> list[str]:
    submission_path = dataset_root / "sample_submission.csv"
    if not submission_path.exists():
        return []
    try:
        columns = pd.read_csv(submission_path, nrows=0).columns.tolist()
    except Exception:
        return []
    return [str(column) for column in columns]


def build_gold_contract(spec: DatasetValidationSpec) -> GoldContract:
    """Build C_gold from a validation-matrix spec + sample_submission columns."""

    dataset_root = (PROJECT_ROOT / spec.dataset).resolve()
    target_columns = normalize_target_columns(spec.target.split(","))
    submission_columns = _resolve_submission_columns(dataset_root)
    target_set = {column.casefold() for column in target_columns}

    # Output columns = submission columns that name the target; the rest are IDs
    # (this mirrors ReaderAgent._submission_spec's role split).
    output_columns = [
        column for column in submission_columns if column.casefold() in target_set
    ]
    id_columns = [
        column for column in submission_columns if column.casefold() not in target_set
    ]
    if not output_columns:
        output_columns = list(target_columns)

    return GoldContract(
        dataset_key=spec.key,
        task=spec.task,
        target_columns=tuple(target_columns),
        task_family=spec.task_family,
        target_mode=spec.target_mode,
        metric_hint=spec.metric_hint,
        dataset_root=dataset_root,
        id_columns=tuple(id_columns),
        output_columns=tuple(output_columns),
        submission_columns=tuple(submission_columns),
    )


def build_registry(keys: Iterable[str] = EVAL_DATASET_KEYS) -> dict[str, GoldContract]:
    specs = _specs()
    missing = [key for key in keys if key not in specs]
    if missing:
        raise KeyError(f"unknown dataset keys: {missing}")
    return {key: build_gold_contract(specs[key]) for key in keys}


__all__ = [
    "EVAL_DATASET_KEYS",
    "GoldContract",
    "OUTPUT_ROOT",
    "PROJECT_ROOT",
    "build_gold_contract",
    "build_registry",
]
