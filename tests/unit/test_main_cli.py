from __future__ import annotations

from pathlib import Path

import pytest

import main as cli
from app.schemas.state_schema import (
    DataOverview,
    EvaluationSpec,
    GraphState,
    ProblemUnderstanding,
    SubmissionSpec,
    TargetSpec,
)


def _reader_update(
    target: str = "Exited", *, target_columns: list[str] | None = None
) -> dict:
    resolved_target_columns = (
        list(target_columns)
        if target_columns is not None
        else ([target] if target else [])
    )
    return {
        "problem_understanding": ProblemUnderstanding(
            task_summary="classification task",
            predicted_task_type="classification",
            target_column="",
            target_columns=[],
            candidate_metrics=["roc_auc"],
            data_overview=DataOverview(column_types={"Exited": "int64"}),
        ).model_dump(mode="python"),
        "target_spec": TargetSpec(
            status="resolved" if resolved_target_columns else "unresolved",
            target_columns=resolved_target_columns,
            target_mode=(
                "single"
                if len(resolved_target_columns) == 1
                else "multi" if len(resolved_target_columns) > 1 else "unknown"
            ),
            task_family="classification",
            prediction_kind="probability" if resolved_target_columns else "",
            submission=SubmissionSpec(
                submission_columns=["id", *resolved_target_columns]
                if resolved_target_columns
                else []
            ),
            evaluation=EvaluationSpec(candidate_metrics=["roc_auc"]),
        ).model_dump(mode="python"),
        "task_config": {
            "target_column": "",
            "target_columns": [],
            "dataset_paths": {
                "train": "train.csv",
                "test": "test.csv",
                "submission": "sample_submission.csv",
            },
        },
        "current_phase": "reader",
        "phase_history": ["reader"],
    }


def _review_update(status: str, *, target_failure: bool = False) -> dict:
    failures = ["target scope is unresolved after Reader"] if target_failure else []
    return {
        "review_state": {
            "latest_review_status": status,
            "review_history": [
                {
                    "phase": "reader",
                    "status": status,
                    "attempt": 1,
                    "blocking_failures": failures,
                    "warnings": [],
                    "checked_artifacts": [],
                    "checked_state_fields": [],
                    "comment": "review",
                    "details": {"train_columns": ["id", "Exited"]},
                }
            ],
            "retry_count_by_phase": {"reader": 1} if status == "retry" else {},
            "fatal_errors": ["fatal"] if status == "fatal" else [],
            "last_review_comment": "review",
            "pending_review_phase": "reader",
        },
        "runtime_state": {"interrupted": status == "fatal", "last_reviewed_phase": "reader"},
    }


def test_print_reader_understanding_uses_english(capsys) -> None:
    state = GraphState()
    state = cli._merge_state(state, _reader_update("Exited"))
    state = cli._merge_state(state, _review_update("approved"))

    cli._print_reader_understanding(state)

    output = capsys.readouterr().out
    assert "Task understanding:" in output
    assert "- Task summary: classification task" in output
    assert "- Target column: Exited" in output
    assert "任务理解" not in output
    assert "目标列" not in output


def test_print_reader_understanding_shows_multi_target_scope(capsys) -> None:
    state = GraphState()
    state = cli._merge_state(
        state,
        _reader_update(
            "",
            target_columns=["Class_1", "Class_2", "Class_3"],
        ),
    )
    state = cli._merge_state(state, _review_update("approved"))

    cli._print_reader_understanding(state)

    output = capsys.readouterr().out
    assert "- Target columns: Class_1, Class_2, Class_3" in output


def test_print_final_paths_deduplicates_equivalent_paths(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    report = tmp_path / "artifacts" / "reports" / "business_report.md"

    cli._print_final_paths(
        {
            "business_report": str(report),
            "business_report_path": "artifacts/reports/business_report.md",
            "prediction_file": "artifacts/predictions/submission.csv",
        }
    )

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert sum(line.startswith("- business_report:") for line in lines) == 1
    assert "business_report_path" not in output
    assert "prediction_file" in output


@pytest.mark.asyncio
async def test_prepare_confirmed_state_auto_yes(monkeypatch) -> None:
    async def fake_reader(state):
        return _reader_update("Exited")

    async def fake_reviewer(state):
        return _review_update("approved")

    monkeypatch.setattr(cli, "run_reader", fake_reader)
    monkeypatch.setattr(cli, "run_bg_reviewer", fake_reviewer)

    result = await cli.prepare_confirmed_state(GraphState(), auto_yes=True)

    assert result is not None
    assert result.target_spec.target_columns == ["Exited"]


@pytest.mark.asyncio
async def test_prepare_confirmed_state_retries_target_hint(monkeypatch) -> None:
    async def fake_reader(state):
        return _reader_update("Exited" if state.task_config.target_hint else "")

    async def fake_reviewer(state):
        if state.target_spec.status == "resolved":
            return _review_update("approved")
        return _review_update("retry", target_failure=True)

    monkeypatch.setattr(cli, "run_reader", fake_reader)
    monkeypatch.setattr(cli, "run_bg_reviewer", fake_reviewer)
    monkeypatch.setattr("builtins.input", lambda _prompt: "Exited")

    result = await cli.prepare_confirmed_state(GraphState(), auto_yes=True)

    assert result is not None
    assert result.task_config.target_hint == "Exited"
    assert result.target_spec.target_columns == ["Exited"]


def test_merge_target_hint_supports_exact_multi_target_input() -> None:
    state = GraphState()
    state.review_state.review_history = [
        {"details": {"train_columns": ["id", "Defect_1", "Defect_2", "feature"]}}
    ]

    updated = cli._merge_target_hint(state, "Defect_1, Defect_2")

    assert updated.task_config.target_column == ""
    assert updated.task_config.target_columns == ["Defect_1", "Defect_2"]


@pytest.mark.asyncio
async def test_prepare_confirmed_state_refusal_exits_cleanly(monkeypatch) -> None:
    async def fake_reader(state):
        return _reader_update("Exited")

    async def fake_reviewer(state):
        return _review_update("approved")

    monkeypatch.setattr(cli, "run_reader", fake_reader)
    monkeypatch.setattr(cli, "run_bg_reviewer", fake_reviewer)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    result = await cli.prepare_confirmed_state(GraphState(), auto_yes=False)

    assert result is None


@pytest.mark.asyncio
async def test_prepare_confirmed_state_fatal_raises(monkeypatch) -> None:
    async def fake_reader(state):
        return _reader_update("Exited")

    async def fake_reviewer(state):
        return _review_update("fatal")

    monkeypatch.setattr(cli, "run_reader", fake_reader)
    monkeypatch.setattr(cli, "run_bg_reviewer", fake_reviewer)

    with pytest.raises(RuntimeError, match="fatal"):
        await cli.prepare_confirmed_state(GraphState(), auto_yes=True)


@pytest.mark.asyncio
async def test_async_main_runs_mocked_workflow(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    async def fake_reader(state):
        return _reader_update("Exited")

    async def fake_reviewer(state):
        return _review_update("approved")

    async def fake_workflow(state, *, recursion_limit):
        assert state.runtime_state.thread_id
        assert recursion_limit == 7
        return {"business_report": "artifacts/reports/business_report.md"}

    monkeypatch.setattr(cli, "configure_runtime", lambda _env_file: None)
    monkeypatch.setattr(cli, "run_reader", fake_reader)
    monkeypatch.setattr(cli, "run_bg_reviewer", fake_reviewer)
    monkeypatch.setattr(cli, "run_full_workflow", fake_workflow)

    code = await cli.async_main(
        [
            "--task",
            "predict churn",
            "--dataset",
            str(dataset_dir),
            "--recursion-limit",
            "7",
            "--yes",
        ]
    )

    assert code == 0
