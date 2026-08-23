"""Unit tests for code execution tooling."""

from __future__ import annotations

import base64
from copy import deepcopy

from app.schemas.artifact_schema import DEFAULT_ARTIFACT_PATHS
from app.schemas.state_schema import GraphState, RuntimeState, TaskConfig
from app.tools.exec_tools import (
    execution_run_id,
    list_execution_run_outputs,
    reset_execution_kernel,
    run_execution,
)

# Minimal valid 1x1 PNG (transparent pixel)
_MIN_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_run_execution_stdout_csv_and_png(tmp_path) -> None:
    code = """
save_execution_csv([["x", "y"], [1, 2]], "sample")
save_execution_png_bytes(_MIN_PNG, "dot")
print("hello")
"""
    code = code.replace("_MIN_PNG", repr(_MIN_PNG))
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[{"role": "code_execution", "code": code}],
    )
    out = run_execution(state)

    assert out["messages"][0]["status"] == "ok"
    assert "hello" in out["messages"][0]["stdout"]
    assert out["messages"][0]["traceback"] is None
    csv_paths = out["messages"][0]["artifacts"]["csv"]
    png_paths = out["messages"][0]["artifacts"]["png"]
    assert any("sample.csv" in p for p in csv_paths)
    assert any("dot.png" in p for p in png_paths)
    assert all(p.startswith("artifacts/execution/") for p in csv_paths + png_paths)
    assert "/step_0001/" in csv_paths[0]
    for rel in csv_paths + png_paths:
        assert (tmp_path / rel).is_file()
    assert png_paths[0] in out["artifacts"]["figures"]
    review = out["messages"][0]["review"]
    assert review["run_id"] == execution_run_id(state)
    assert review["step"] == 1
    assert review["notebook_id"] == 0
    assert len(review["files"]) >= 2


def test_execution_run_id_prefers_task_id_over_thread_id(tmp_path) -> None:
    state = GraphState(
        task_config=TaskConfig(task_id="task-from-config"),
        runtime_state=RuntimeState(workspace_root=str(tmp_path), thread_id="thread-99"),
        messages=[{"role": "code_execution", "code": "pass"}],
    )
    assert execution_run_id(state) == "task-from-config"
    out = run_execution(state)
    assert out["messages"][0]["execution_run_id"] == "task-from-config"
    assert "artifacts/execution/task-from-config/" in out["messages"][0]["review"]["output_root_relative"]


def test_save_markdown_and_list_run_outputs(tmp_path) -> None:
    tid = "review-run-alpha"
    rt = RuntimeState(workspace_root=str(tmp_path), thread_id=tid)
    state = GraphState(
        runtime_state=rt,
        messages=[
            {
                "role": "code_execution",
                "code": 'save_execution_markdown("# Hi\\n", "note")',
            }
        ],
    )
    out = run_execution(state)
    assert out["messages"][0]["artifacts"]["markdown"]
    md_rel = out["messages"][0]["artifacts"]["markdown"][0]
    assert md_rel.endswith("note.md")
    assert (tmp_path / md_rel).read_text(encoding="utf-8").startswith("# Hi")

    idx = list_execution_run_outputs(tmp_path, run_id=tid)
    assert idx["run_id"] == tid
    kinds = {f["kind"] for f in idx["files"]}
    assert "markdown" in kinds


def test_list_execution_run_outputs_missing_run(tmp_path) -> None:
    (tmp_path / "artifacts" / "execution").mkdir(parents=True)
    idx = list_execution_run_outputs(tmp_path, run_id="does-not-exist")
    assert idx.get("error") == "run_not_found"
    assert idx["files"] == []


def test_list_all_run_ids(tmp_path) -> None:
    (tmp_path / "artifacts" / "execution" / "run_a" / "nb_0").mkdir(parents=True)
    (tmp_path / "artifacts" / "execution" / "run_b" / "nb_0").mkdir(parents=True)
    idx = list_execution_run_outputs(tmp_path, run_id=None)
    assert set(idx["run_ids"]) == {"run_a", "run_b"}


def test_list_run_ids_empty_when_no_execution_root(tmp_path) -> None:
    idx = list_execution_run_outputs(tmp_path / "no_artifacts_yet", run_id=None)
    assert idx["run_ids"] == []


def test_run_execution_accepts_agent_assistant_content(tmp_path) -> None:
    """LLM/agent output often uses role=assistant and code in content."""
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[{"role": "assistant", "content": 'print("from_agent")'}],
    )
    out = run_execution(state)
    assert out["messages"][0]["status"] == "ok"
    assert "from_agent" in out["messages"][0]["stdout"]


def test_run_execution_developer_role_content(tmp_path) -> None:
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[{"role": "developer", "content": "y = 100\nprint(y)"}],
    )
    out = run_execution(state)
    assert out["messages"][0]["status"] == "ok"
    assert "100" in out["messages"][0]["stdout"]


def test_explicit_code_key_over_content(tmp_path) -> None:
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[
            {
                "role": "assistant",
                "content": "print('wrong')",
                "code": "print('right')",
            }
        ],
    )
    out = run_execution(state)
    assert "right" in out["messages"][0]["stdout"]
    assert "wrong" not in out["messages"][0]["stdout"]


def test_stderr_captured(tmp_path) -> None:
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[
            {
                "role": "code_execution",
                "code": "import sys\nsys.stderr.write('errline\\n')",
            }
        ],
    )
    out = run_execution(state)
    assert "errline" in out["messages"][0]["stderr"]


def test_run_execution_no_messages(tmp_path) -> None:
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[],
    )
    out = run_execution(state)
    assert out["messages"][0]["status"] == "error"
    assert "error" in out["messages"][0]


def test_run_execution_empty_code_message(tmp_path) -> None:
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[{"role": "code_execution", "code": "   \n"}],
    )
    out = run_execution(state)
    assert out["messages"][0]["status"] == "error"


def test_multistep_execution_preserves_variables(tmp_path, unique_thread_id: str) -> None:
    """Sequential runs share one kernel per thread (Jupyter-like)."""
    rt = RuntimeState(
        workspace_root=str(tmp_path),
        thread_id=unique_thread_id,
        checkpoint_namespace="default",
    )
    s1 = GraphState(
        runtime_state=rt,
        messages=[{"role": "code_execution", "code": "x = 40"}],
    )
    out1 = run_execution(s1)
    assert out1["messages"][0]["sequential_step"] == 1
    assert out1["messages"][0]["status"] == "ok"

    s2 = GraphState(
        runtime_state=rt,
        messages=[{"role": "code_execution", "code": "print(x + 2)"}],
    )
    out2 = run_execution(s2)
    assert out2["messages"][0]["sequential_step"] == 2
    assert "42" in out2["messages"][0]["stdout"]

    reset_execution_kernel(s2)
    s3 = GraphState(
        runtime_state=rt,
        messages=[{"role": "code_execution", "code": "print(x)"}],
    )
    out3 = run_execution(s3)
    assert out3["messages"][0]["sequential_step"] == 1
    assert out3["messages"][0]["status"] == "error"
    assert out3["messages"][0]["review"]["notebook_id"] == 1


def test_different_threads_do_not_share_variables(tmp_path) -> None:
    a = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path), thread_id="t-a"),
        messages=[{"role": "code_execution", "code": "z = 1"}],
    )
    run_execution(a)
    b = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path), thread_id="t-b"),
        messages=[{"role": "code_execution", "code": "print(z)"}],
    )
    out = run_execution(b)
    assert out["messages"][0]["status"] == "error"
    assert "z" in out["messages"][0]["traceback"] or "name" in out["messages"][0]["traceback"].lower()


def test_checkpoint_namespace_isolates_kernels(tmp_path) -> None:
    """Same thread_id but different checkpoint_namespace -> different kernel."""
    rt1 = RuntimeState(
        workspace_root=str(tmp_path),
        thread_id="same-tid",
        checkpoint_namespace="ns-one",
    )
    rt2 = RuntimeState(
        workspace_root=str(tmp_path),
        thread_id="same-tid",
        checkpoint_namespace="ns-two",
    )
    run_execution(
        GraphState(
            runtime_state=rt1,
            messages=[{"role": "code_execution", "code": "u = 'one'"}],
        )
    )
    out = run_execution(
        GraphState(
            runtime_state=rt2,
            messages=[{"role": "code_execution", "code": "print(u)"}],
        )
    )
    assert out["messages"][0]["status"] == "error"


def test_run_execution_merges_figures_preserves_other_artifact_keys(tmp_path) -> None:
    arts = deepcopy(DEFAULT_ARTIFACT_PATHS)
    arts["technical_report"] = "artifacts/reports/technical_report.md"
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[{"role": "code_execution", "code": f"save_execution_png_bytes({repr(_MIN_PNG)}, 'x')"}],
        artifacts=arts,
    )
    out = run_execution(state)
    assert out["artifacts"]["technical_report"] == "artifacts/reports/technical_report.md"
    assert out["artifacts"]["figures"]


def test_run_execution_syntax_error_records_traceback(tmp_path) -> None:
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[{"role": "code_execution", "code": "def oops("}],
    )
    out = run_execution(state)
    assert out["messages"][0]["status"] == "error"
    assert out["messages"][0]["traceback"] is not None
    assert "error" in out["messages"][0]


def test_runtime_error_in_user_code(tmp_path) -> None:
    state = GraphState(
        runtime_state=RuntimeState(workspace_root=str(tmp_path)),
        messages=[{"role": "code_execution", "code": "1 / 0"}],
    )
    out = run_execution(state)
    assert out["messages"][0]["status"] == "error"
    assert "ZeroDivision" in (out["messages"][0]["traceback"] or "")
    assert out["messages"][0]["error"]["error_type"] == "runtime_error"
