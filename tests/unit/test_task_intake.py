from __future__ import annotations

from pathlib import Path

import pytest

from app.intake import build_initial_state
def test_task_intake_preserves_directory_input(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    state = build_initial_state(
        task="predict churn",
        dataset=str(dataset_dir),
        workspace_root=str(tmp_path),
        thread_id="run-1",
    )

    assert state.task_config.raw_task == "predict churn"
    assert state.task_config.business_context == "predict churn"
    assert state.task_config.dataset_root == str(dataset_dir.resolve())
    assert state.task_config.dataset_paths.train == ""
    assert state.runtime_state.thread_id == "run-1"
    assert state.runtime_state.run_id.startswith("run-1-run-")
    assert state.task_config.task_id == state.runtime_state.run_id
    assert state.runtime_state.artifact_root.startswith("artifacts/runs/dataset_")
    assert state.runtime_state.artifact_root.endswith(state.runtime_state.run_id[-8:])
    assert state.artifacts["technical_report"].startswith(
        f"{state.runtime_state.artifact_root}/reports/"
    )
    assert state.runtime_state.workspace_root == str(tmp_path.resolve())


def test_task_intake_preserves_optional_target_hint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    state = build_initial_state(
        task="predict customer churn",
        dataset=str(dataset_dir),
        target="customer churn",
        workspace_root=str(tmp_path),
    )

    assert state.task_config.target_hint == "customer churn"
    assert state.task_config.target_column == ""
    assert state.task_config.target_columns == []
    assert state.runtime_state.thread_id.startswith("cli-")
    assert state.runtime_state.run_id.startswith(f"{state.runtime_state.thread_id}-run-")


def test_task_intake_accepts_file_as_train_path(tmp_path: Path) -> None:
    dataset_file = tmp_path / "train.csv"
    dataset_file.write_text("target,x\n1,2\n", encoding="utf-8")

    state = build_initial_state(
        task="predict target",
        dataset=str(dataset_file),
        target="target",
        workspace_root=str(tmp_path),
    )

    assert state.task_config.dataset_root == str(dataset_file.resolve())
    assert state.task_config.dataset_paths.train == str(dataset_file.resolve())
    assert state.task_config.target_column == "target"
    assert state.task_config.target_columns == ["target"]


def test_task_intake_preserves_metric_hint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    state = build_initial_state(
        task="predict target",
        dataset=str(dataset_dir),
        metric_hint="f1_macro",
        workspace_root=str(tmp_path),
    )

    assert state.task_config.metric_hint == "f1_macro"


def test_task_intake_allocates_unique_artifact_roots_per_run(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    first = build_initial_state(
        task="predict churn",
        dataset=str(dataset_dir),
        workspace_root=str(tmp_path),
        thread_id="repeated-task",
    )
    second = build_initial_state(
        task="predict churn",
        dataset=str(dataset_dir),
        workspace_root=str(tmp_path),
        thread_id="repeated-task",
    )

    assert first.runtime_state.thread_id == second.runtime_state.thread_id == "repeated-task"
    assert first.runtime_state.run_id != second.runtime_state.run_id
    assert first.runtime_state.artifact_root != second.runtime_state.artifact_root
    assert first.artifacts["feature_summary"] != second.artifacts["feature_summary"]


def test_task_intake_rejects_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_initial_state(
            task="predict target",
            dataset=str(tmp_path / "missing"),
            workspace_root=str(tmp_path),
        )


def test_task_intake_rejects_blank_task(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    with pytest.raises(ValueError):
        build_initial_state(task="", dataset=str(dataset_dir), workspace_root=str(tmp_path))
