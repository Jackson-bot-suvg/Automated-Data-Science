"""Python code execution: Jupyter-like kernels, run-scoped CSV/PNG/Markdown outputs, review index."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.schemas.state_schema import GraphState

_AGENT_CODE_ROLES = frozenset({"code_execution", "developer", "assistant"})

# In-process execution namespaces (variables survive sequential runs; not checkpoint-restored).
_SESSION_NAMESPACES: dict[str, dict[str, Any]] = {}
_SESSION_STEPS: dict[str, int] = {}
# After reset_execution_kernel, notebook increments so outputs stay in separate folders.
_SESSION_NOTEBOOK: dict[str, int] = {}

_EXEC_ROOT = Path("artifacts") / "execution"


def _slug(value: str, max_len: int = 64) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    if not s:
        return "run"
    return s[:max_len]


def execution_run_id(state: GraphState) -> str:
    """Stable id for organizing outputs (task id, else thread id, else workspace hash)."""
    task = (state.task_config.task_id or "").strip()
    if task:
        return _slug(task)
    tid = (state.runtime_state.thread_id or "").strip()
    if tid:
        return _slug(tid)
    ws = str(Path(state.runtime_state.workspace_root or ".").resolve())
    h = hashlib.sha256(ws.encode("utf-8")).hexdigest()[:12]
    return f"ephemeral_{h}"


def _session_key(state: GraphState) -> str:
    rt = state.runtime_state
    tid = (rt.thread_id or "").strip()
    ns = (rt.checkpoint_namespace or "default").strip()
    if tid:
        return f"{ns}:{tid}"
    ws = Path(rt.workspace_root or ".").resolve()
    return f"_ephemeral:{ws}"


def _notebook_id(session_key: str) -> int:
    return _SESSION_NOTEBOOK.get(session_key, 0)


def reset_execution_kernel(state: GraphState) -> None:
    """Clear the in-memory kernel; next runs use a new notebook folder under the same run id."""
    key = _session_key(state)
    _SESSION_NAMESPACES.pop(key, None)
    _SESSION_STEPS.pop(key, None)
    _SESSION_NOTEBOOK[key] = _SESSION_NOTEBOOK.get(key, 0) + 1


def _step_output_prefix(run_id: str, notebook_id: int, step: int) -> str:
    return (_EXEC_ROOT / run_id / f"nb_{notebook_id}" / f"step_{step:04d}").as_posix()


def _inject_helpers(
    namespace: dict[str, Any],
    workspace: Path,
    audit: dict[str, list[str]],
    *,
    run_id: str,
    notebook_id: int,
    step: int,
) -> None:
    """Bind save_* helpers; paths live under artifacts/execution/<run_id>/nb_*/step_*/."""
    figures_rel: list[str] = []
    csv_rel: list[str] = []
    md_rel: list[str] = []
    audit["png"] = figures_rel
    audit["csv"] = csv_rel
    audit["markdown"] = md_rel

    base = _step_output_prefix(run_id, notebook_id, step)

    def _safe_name(name: str, default: str) -> str:
        s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
        return s if s else default

    def save_execution_csv(rows: list[list[Any]], name: str) -> str:
        safe = _safe_name(name, "output")
        rel = f"{base}/data/{safe}.csv"
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        csv_rel.append(rel)
        return rel

    def save_execution_png_bytes(data: bytes, name: str) -> str:
        safe = _safe_name(name, "figure")
        rel = f"{base}/figures/{safe}.png"
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        figures_rel.append(rel)
        return rel

    def save_execution_figure(fig: Any, name: str) -> str:
        try:
            import matplotlib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for save_execution_figure") from exc
        safe = _safe_name(name, "figure")
        rel = f"{base}/figures/{safe}.png"
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        matplotlib.use("Agg")
        fig.savefig(path, format="png", bbox_inches="tight")
        figures_rel.append(rel)
        return rel

    def save_execution_markdown(content: str, name: str) -> str:
        safe = _safe_name(name, "report")
        rel = f"{base}/reports/{safe}.md"
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        md_rel.append(rel)
        return rel

    namespace["save_execution_csv"] = save_execution_csv
    namespace["save_execution_png_bytes"] = save_execution_png_bytes
    namespace["save_execution_figure"] = save_execution_figure
    namespace["save_execution_markdown"] = save_execution_markdown


def _get_or_create_namespace(
    session_key: str,
    workspace: Path,
    audit: dict[str, list[str]],
    *,
    run_id: str,
    notebook_id: int,
    step: int,
) -> dict[str, Any]:
    existing = _SESSION_NAMESPACES.get(session_key)
    prev_ws = existing.get("_exec_workspace") if existing else None
    if existing is None:
        existing = {"__builtins__": __builtins__}
        _SESSION_NAMESPACES[session_key] = existing
    elif prev_ws != workspace:
        # Workspace moved: new kernel semantics; step counter is reset in run_execution path via pop below
        _SESSION_STEPS.pop(session_key, None)
        existing = {"__builtins__": __builtins__}
        _SESSION_NAMESPACES[session_key] = existing
    existing["_exec_workspace"] = workspace
    _inject_helpers(existing, workspace, audit, run_id=run_id, notebook_id=notebook_id, step=step)
    return existing


def _build_review_file_index(audit: dict[str, list[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rel in audit.get("csv", []):
        rows.append({"kind": "csv", "path": rel})
    for rel in audit.get("png", []):
        rows.append({"kind": "png", "path": rel})
    for rel in audit.get("markdown", []):
        rows.append({"kind": "markdown", "path": rel})
    return rows


def list_execution_run_outputs(workspace_root: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """Scan disk for saved execution outputs (for human or reviewer tooling).

    If ``run_id`` is None, returns an index of all run ids found under ``artifacts/execution/``.
    Otherwise returns files grouped by kind under that run id.
    """
    root = Path(workspace_root).resolve() / _EXEC_ROOT
    if not root.is_dir():
        return {"run_ids": [], "runs": {}}

    if run_id is None:
        run_ids = sorted(p.name for p in root.iterdir() if p.is_dir())
        return {"run_ids": run_ids, "runs": {}}

    run_dir = root / run_id
    if not run_dir.is_dir():
        return {"run_id": run_id, "error": "run_not_found", "files": []}

    files: list[dict[str, str]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(Path(workspace_root).resolve())
        suf = path.suffix.lower()
        if suf == ".csv":
            kind = "csv"
        elif suf == ".png":
            kind = "png"
        elif suf == ".md":
            kind = "markdown"
        else:
            kind = "other"
        files.append({"kind": kind, "path": str(rel).replace("\\", "/")})
    return {"run_id": run_id, "files": files}


def _extract_code(state: GraphState) -> str:
    msgs = state.messages
    if not msgs:
        return ""
    last = msgs[-1]
    raw = last.get("code")
    if isinstance(raw, str) and raw.strip():
        return raw
    role = str(last.get("role", ""))
    if role in _AGENT_CODE_ROLES and isinstance(last.get("content"), str):
        body = last["content"]
        return body if body.strip() else ""
    return ""


def run_execution(state: GraphState) -> dict:
    """Execute Python from the last message; persist outputs under ``artifacts/execution/<run_id>/``."""
    code = _extract_code(state)
    workspace = Path(state.runtime_state.workspace_root or ".").resolve()
    session_key = _session_key(state)
    run_id = execution_run_id(state)
    nb_id = _notebook_id(session_key)

    audit: dict[str, list[str]] = {"png": [], "csv": [], "markdown": []}
    stdout_text = ""
    stderr_text = ""
    tb: str | None = None
    status = "ok"
    err_message = ""

    if not code.strip():
        status = "error"
        err_message = (
            "No executable code in messages (expected last message with non-empty 'code', "
            "or role assistant/developer/code_execution with non-empty 'content')."
        )
        step_guess = _SESSION_STEPS.get(session_key, 0)
        record: dict[str, Any] = {
            "role": "code_execution",
            "source": "exec_tools.run_execution",
            "status": status,
            "stdout": "",
            "stderr": "",
            "traceback": None,
            "execution_run_id": run_id,
            "artifacts": {"csv": [], "png": [], "markdown": []},
            "sequential_step": step_guess,
            "kernel_session": session_key,
            "review": {
                "run_id": run_id,
                "notebook_id": nb_id,
                "step": step_guess,
                "output_root_relative": "",
                "files": [],
            },
            "error": {
                "status": "error",
                "phase": "code_execution",
                "error_type": "schema_error",
                "message": err_message,
                "details": {"traceback": None, "context": {}},
            },
        }
        arts = deepcopy(dict(state.artifacts))
        return {"messages": [record], "artifacts": arts}

    step = _SESSION_STEPS.get(session_key, 0) + 1
    _SESSION_STEPS[session_key] = step

    out_rel = _step_output_prefix(run_id, nb_id, step)

    namespace = _get_or_create_namespace(
        session_key,
        workspace,
        audit,
        run_id=run_id,
        notebook_id=nb_id,
        step=step,
    )
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<code_execution>", "exec"), namespace, namespace)
    except Exception as exc:  # noqa: BLE001 — execution sandbox reports any failure
        status = "error"
        tb = traceback.format_exc()
        err_message = str(exc)
    stdout_text = stdout_buf.getvalue()
    stderr_text = stderr_buf.getvalue()

    file_index = _build_review_file_index(audit)
    record = {
        "role": "code_execution",
        "source": "exec_tools.run_execution",
        "status": status,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "traceback": tb,
        "execution_run_id": run_id,
        "artifacts": {"csv": audit["csv"], "png": audit["png"], "markdown": audit["markdown"]},
        "sequential_step": step,
        "kernel_session": session_key,
        "review": {
            "run_id": run_id,
            "notebook_id": nb_id,
            "step": step,
            "output_root_relative": out_rel,
            "files": file_index,
        },
    }
    if status == "error":
        record["error"] = {
            "status": "error",
            "phase": "code_execution",
            "error_type": "runtime_error",
            "message": err_message or "execution failed",
            "details": {"traceback": tb, "context": {}},
        }

    arts = deepcopy(dict(state.artifacts))
    existing_figures = list(arts.get("figures") or [])
    arts["figures"] = existing_figures + audit["png"]

    return {"messages": [record], "artifacts": arts}
