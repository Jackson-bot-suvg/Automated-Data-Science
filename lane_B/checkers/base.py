"""Shared data structures and deterministic contract logic for the checkers.

The intake resolution reuses the project's real deterministic logic in
``app.agents.reader.ReaderAgent`` (no LLM is invoked): it resolves the target
from train/test column differences, the sample submission, the overview text,
and the natural-language task, then splits the submission into id/output roles.

The post-intake checks mirror the semantic parts of
``app.tools.reviewer_tools`` (target leakage, prediction-vs-submission contract,
row-count alignment). Column comparison is set-based (order-insensitive), which
is the intended submission semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.agents.reader import ReaderAgent
from app.schemas.state_schema import DatasetPaths, GraphState
from app.tools import reviewer_tools
from app.utils.target_scope import exact_target_matches, normalize_target_columns

from lane_B.registry import PROJECT_ROOT


@dataclass
class Judgment:
    method_id: str  # "M1" | "M2" | "F0"
    verdict: str  # "pass" | "block"
    root_causes: list[str] = field(default_factory=list)
    earliest_stage: str = "none"  # "none" | "intake" | "post_intake"
    details: dict = field(default_factory=dict)


@dataclass
class VisiblePackage:
    """The agent-visible materials for one mutated case (no ground truth)."""

    data_dir: Path
    task_text: str
    overview_text: str
    train_columns: list[str]
    test_columns: list[str]
    submission_columns: list[str]
    prediction_columns: list[str]
    test_rows: int
    prediction_rows: int
    has_prediction: bool


@dataclass
class IntakeResolution:
    """The contract resolved at intake from visible materials (C_read)."""

    status: str  # "resolved" | "unresolved"
    target_columns: list[str]
    id_columns: list[str]
    output_columns: list[str]
    submission_columns: list[str]
    named_targets_in_text: list[str]
    source: str


def load_package(data_dir: Path) -> VisiblePackage:
    """Read the visible package files and return their structural metadata."""

    workspace = str(PROJECT_ROOT)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    submission_path = data_dir / "sample_submission.csv"
    prediction_path = data_dir / "prediction.csv"
    overview_path = data_dir / "overview.txt"

    task_text = (data_dir / "task.txt").read_text(encoding="utf-8") if (data_dir / "task.txt").exists() else ""
    overview_text = overview_path.read_text(encoding="utf-8", errors="ignore") if overview_path.exists() else ""

    train_columns = reviewer_tools.csv_columns(workspace, str(train_path))
    test_columns = reviewer_tools.csv_columns(workspace, str(test_path))
    submission_columns = reviewer_tools.csv_columns(workspace, str(submission_path))
    has_prediction = prediction_path.exists()
    prediction_columns = reviewer_tools.csv_columns(workspace, str(prediction_path)) if has_prediction else []

    test_shape = reviewer_tools.csv_shape(workspace, str(test_path))
    prediction_shape = reviewer_tools.csv_shape(workspace, str(prediction_path)) if has_prediction else None

    return VisiblePackage(
        data_dir=data_dir,
        task_text=task_text,
        overview_text=overview_text,
        train_columns=train_columns,
        test_columns=test_columns,
        submission_columns=submission_columns,
        prediction_columns=prediction_columns,
        test_rows=int(test_shape[0]) if test_shape else 0,
        prediction_rows=int(prediction_shape[0]) if prediction_shape else 0,
        has_prediction=has_prediction,
    )


def resolve_intake(package: VisiblePackage) -> IntakeResolution:
    """Resolve C_read at intake using the real ReaderAgent deterministic logic."""

    reader = ReaderAgent()
    train_meta = reader._csv_metadata(package.data_dir / "train.csv")
    test_meta = reader._csv_metadata(package.data_dir / "test.csv")
    submission_meta = reader._csv_metadata(package.data_dir / "sample_submission.csv")

    state = GraphState()
    state.task_config.raw_task = package.task_text
    state.task_config.target_hint = ""
    state.task_config.target_column = ""
    state.task_config.target_columns = []
    state.task_config.dataset_paths = DatasetPaths(
        train=str(package.data_dir / "train.csv"),
        test=str(package.data_dir / "test.csv"),
        submission=str(package.data_dir / "sample_submission.csv"),
    )
    state.runtime_state.workspace_root = str(PROJECT_ROOT)

    resolved = reader._resolve_target_scope(
        state, train_meta, test_meta, submission_meta, package.overview_text
    )
    target_columns = normalize_target_columns(resolved.get("target_columns") or [])
    submission_spec = reader._submission_spec(target_columns, submission_meta)
    source = ""
    evidence = resolved.get("evidence") or []
    if evidence:
        source = str(evidence[0].get("source", "")) if isinstance(evidence[0], dict) else str(getattr(evidence[0], "source", ""))

    named = exact_target_matches(package.task_text, package.train_columns)

    return IntakeResolution(
        status=str(resolved.get("status", "unresolved")),
        target_columns=target_columns,
        id_columns=normalize_target_columns(submission_spec.get("id_columns") or []),
        output_columns=normalize_target_columns(submission_spec.get("output_columns") or []),
        submission_columns=normalize_target_columns(submission_spec.get("submission_columns") or []),
        named_targets_in_text=named,
        source=source,
    )


def intake_semantic_checks(
    resolution: IntakeResolution, package: VisiblePackage
) -> list[tuple[str, str]]:
    """Return (fault_field, reason) pairs for entry-stage semantic violations."""

    causes: list[tuple[str, str]] = []
    if resolution.status != "resolved" or not resolution.target_columns:
        causes.append(("target_columns", "target scope unresolved from visible materials"))
        return causes

    missing = [c for c in resolution.target_columns if c not in package.train_columns]
    if missing:
        causes.append(("target_columns", "resolved target missing from train: " + ", ".join(missing)))

    leaked = [c for c in resolution.target_columns if c in package.test_columns]
    if leaked:
        causes.append(("target_columns", "target leaked into test: " + ", ".join(leaked)))

    if not resolution.output_columns:
        causes.append(
            ("output_columns", "sample submission has no output column matching the resolved target")
        )

    named = resolution.named_targets_in_text
    if named and set(named) != set(resolution.target_columns):
        causes.append(
            ("target_columns", "task text names a different target than the data: " + ", ".join(named))
        )

    return causes


def post_intake_checks(
    resolution: IntakeResolution, package: VisiblePackage
) -> list[tuple[str, str]]:
    """Return (fault_field, reason) pairs for post-intake (later-stage) drift."""

    causes: list[tuple[str, str]] = []
    if not resolution.target_columns:
        return causes

    # Re-validate target leakage at a later stage (mirrors reviewer_tools).
    leaked = [c for c in resolution.target_columns if c in package.test_columns]
    if leaked:
        causes.append(("target_columns", "target leaked into test: " + ", ".join(leaked)))

    # Prediction contract check (mirrors _check_prediction_columns_and_rows).
    if package.has_prediction and resolution.submission_columns:
        if set(package.prediction_columns) != set(resolution.submission_columns):
            causes.append(
                (
                    "output_columns",
                    "prediction columns do not match submission contract; "
                    f"expected={resolution.submission_columns}, actual={package.prediction_columns}",
                )
            )
        elif package.prediction_rows != package.test_rows:
            causes.append(
                (
                    "output_columns",
                    f"prediction row count {package.prediction_rows} != test row count {package.test_rows}",
                )
            )

    return causes


def _judgment(method_id: str, causes: list[tuple[str, str]], stage: str) -> Judgment:
    if not causes:
        return Judgment(method_id=method_id, verdict="pass", earliest_stage="none")
    fields = list({field for field, _ in causes})
    reasons = [reason for _, reason in causes]
    return Judgment(
        method_id=method_id,
        verdict="block",
        root_causes=fields,
        earliest_stage=stage,
        details={"reasons": reasons},
    )


__all__ = [
    "IntakeResolution",
    "Judgment",
    "VisiblePackage",
    "intake_semantic_checks",
    "load_package",
    "post_intake_checks",
    "resolve_intake",
]
