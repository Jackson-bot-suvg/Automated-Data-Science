"""Filesystem helpers shared by agent artifact writers."""

from __future__ import annotations

from pathlib import Path


def artifact_root_path(workspace_root: str, artifact_root: str) -> Path:
    """Resolve the per-run artifact root under the workspace."""

    root = Path(artifact_root or "artifacts")
    if root.is_absolute():
        return root
    return Path(workspace_root or ".").joinpath(root).resolve()


def figure_path(
    *,
    workspace_root: str,
    artifact_root: str,
    filename: str,
) -> str:
    """Build a figure path under the per-run figures directory."""

    root = artifact_root_path(workspace_root, artifact_root)
    target = root / "figures" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(target)


def ensure_artifact_parent(path: str | Path) -> Path:
    """Ensure the parent directory exists and return the normalized path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
