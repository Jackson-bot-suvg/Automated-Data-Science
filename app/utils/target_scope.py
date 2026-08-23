"""Helpers for keeping single-target and multi-target scope consistent."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from app.schemas.state_schema import GraphState


def normalize_target_columns(columns: Iterable[str] | None) -> list[str]:
    """Return de-duplicated target columns while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in columns or []:
        column = str(raw or "").strip()
        if not column:
            continue
        key = column.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(column)
    return normalized


def canonical_target_scope(
    target_column: str = "", target_columns: Iterable[str] | None = None
) -> tuple[str, list[str]]:
    """Canonicalize target scope so single-target paths stay backward compatible."""

    single = str(target_column or "").strip()
    columns = normalize_target_columns(target_columns)
    if not columns and single:
        columns = [single]
    resolved_single = columns[0] if len(columns) == 1 else ""
    return resolved_single, columns


def exact_target_matches(text: str, train_columns: Iterable[str] | None) -> list[str]:
    """Return exact target column mentions from free-form text."""

    hint = str(text or "").strip().lower()
    if not hint:
        return []

    matches: list[str] = []
    for column in normalize_target_columns(train_columns):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(column.lower())}(?![A-Za-z0-9_])"
        if re.search(pattern, hint):
            matches.append(column)
    return matches


def get_target_columns(state: GraphState) -> list[str]:
    """从 GraphState 获取目标列，优先读 target_spec，fallback 到旧字段。"""

    # 优先：target_spec.target_columns
    if state.target_spec.target_columns:
        return list(state.target_spec.target_columns)

    # Fallback 1: task_config.target_columns
    if state.task_config.target_columns:
        return list(state.task_config.target_columns)

    # Fallback 2: task_config.target_column (单目标旧字段)
    if state.task_config.target_column:
        return [state.task_config.target_column]

    # Fallback 3: data_state.target_columns
    if state.data_state.target_columns:
        return list(state.data_state.target_columns)

    # Fallback 4: data_state.target_column
    if state.data_state.target_column:
        return [state.data_state.target_column]

    # Fallback 5: problem_understanding.target_columns
    if state.problem_understanding.target_columns:
        return list(state.problem_understanding.target_columns)

    if state.problem_understanding.target_column:
        return [state.problem_understanding.target_column]

    return []


def get_id_columns(state: GraphState) -> list[str]:
    """从 GraphState 获取 ID 列。"""

    if state.target_spec.submission.id_columns:
        return list(state.target_spec.submission.id_columns)
    return []


