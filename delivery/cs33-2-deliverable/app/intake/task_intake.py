"""CLI task intake helpers.

TaskIntake intentionally stays thin: it validates the user supplied task and
dataset location, then stores the original input in ``task_config`` for Reader.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.schemas.artifact_schema import build_run_artifact_paths, build_run_artifact_root
from app.schemas.state_schema import DatasetPaths, GraphState, RuntimeState, TaskConfig
from app.utils.target_scope import canonical_target_scope


def build_initial_state(
    *,
    task: str,
    dataset: str,
    target: str = "",
    metric_hint: str = "",
    workspace_root: str = ".",
    thread_id: str = "",
) -> GraphState:
    """Build the initial ``GraphState`` from CLI input."""

    raw_task = _required_text(task, "task")
    dataset_input = _required_text(dataset, "dataset")
    root = Path(workspace_root or ".").resolve()
    dataset_path = _resolve(root, dataset_input)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not (dataset_path.is_dir() or dataset_path.is_file()):
        raise ValueError(f"Dataset path must be a directory or file: {dataset_path}")

    dataset_root = str(dataset_path)
    dataset_paths = DatasetPaths()
    if dataset_path.is_file():
        dataset_paths.train = str(dataset_path)

    logical_thread_id = thread_id.strip() if thread_id.strip() else f"cli-{uuid4().hex[:8]}"
    run_id = f"{logical_thread_id}-run-{uuid4().hex[:8]}"
    dataset_label = _dataset_label(dataset_path)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_run_id = run_id[-8:]
    run_label = f"{dataset_label}_{run_timestamp}_{short_run_id}"
    artifact_root = build_run_artifact_root(run_id, run_label=run_label)
    target_hint = target.strip()
    resolved_metric_hint = metric_hint.strip()
    initial_target_column, initial_target_columns = canonical_target_scope(
        target_column=target_hint if _looks_like_column_name(target_hint) else ""
    )
    return GraphState(
        task_config=TaskConfig(
            task_id=run_id,
            raw_task=raw_task,
            dataset_root=dataset_root,
            target_hint=target_hint,
            metric_hint=resolved_metric_hint,
            dataset_paths=dataset_paths,
            business_context=raw_task,
            target_column=initial_target_column,
            target_columns=initial_target_columns,
        ),
        artifacts=build_run_artifact_paths(artifact_root),
        runtime_state=RuntimeState(
            thread_id=logical_thread_id,
            run_id=run_id,
            artifact_root=artifact_root,
            workspace_root=str(root),
        ),
    )


def _required_text(value: str, field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _resolve(workspace_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _looks_like_column_name(value: str) -> bool:
    if not value:
        return False
    return all(ch.isalnum() or ch == "_" for ch in value)


def _dataset_label(dataset_path: Path) -> str:
    if dataset_path.is_dir():
        raw = dataset_path.name
    elif dataset_path.parent and dataset_path.parent.name:
        raw = dataset_path.parent.name
    else:
        raw = dataset_path.stem
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw.strip().lower())
    return cleaned or "dataset"
