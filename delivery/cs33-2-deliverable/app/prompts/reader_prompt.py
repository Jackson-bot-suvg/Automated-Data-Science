"""Prompt assembly for the main-chain Reader agent."""

from __future__ import annotations

import json
from typing import Any


READER_SYSTEM_PROMPT = """
You are the Reader agent for a data-science LangGraph workflow.
Use only the bounded task and dataset metadata provided by the user prompt.
Return a single JSON object. Do not include markdown.
Keep deterministic metadata authoritative; improve wording and risk framing only when evidence supports it.
"""


def build_reader_prompt(*, reader_context: dict[str, Any], fallback: dict[str, Any]) -> str:
    """Build a bounded Reader prompt from task configuration and CSV metadata."""

    return "\n".join(
        [
            "You are reading task configuration and dataset metadata to initialize the workflow.",
            "Create structured problem understanding from the provided context.",
            "",
            "Return JSON with these keys:",
            "- task_summary: string (1-2 sentences, e.g., 'Binary classification of Titanic survival')",
            "- predicted_task_type: string ('binary_classification', 'multiclass_classification', 'multilabel_classification', or 'regression')",
            "- target_column: string (exact column name, case-sensitive)",
            "- target_columns: list of strings (exact column names; keep the full set when multi-target)",
            "- candidate_metrics: list of strings using canonical metric names for the inferred task family",
            "- risks: list of strings (e.g., 'Missing values in Age', 'Small dataset')",
            "- business_goal_summary: string (2-3 sentences from task_config.business_context)",
            "",
            "Instructions:",
            "1. Use train_metadata.columns to validate target_column and target_columns.",
            "2. Keep deterministic target evidence authoritative; do not collapse valid multi-target scope to one column.",
            "3. Infer canonical task family from target scope cardinality and dtype if not explicit.",
            "4. List data quality risks: missing values, class imbalance, small size, etc.",
            "5. Do not invent metrics or risks not present in the metadata.",
            "",
            "Bounded reader context:",
            json.dumps(reader_context, ensure_ascii=False, indent=2, default=str),
            "",
            "Deterministic fallback payload (use if uncertain):",
            json.dumps(fallback, ensure_ascii=False, indent=2, default=str),
        ]
    )


__all__ = ["READER_SYSTEM_PROMPT", "build_reader_prompt"]
