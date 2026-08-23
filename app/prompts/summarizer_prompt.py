"""Prompt assembly for the Technical Summarizer agent."""

from __future__ import annotations

import json
from typing import Any


SUMMARIZER_SYSTEM_PROMPT = """
You are the Technical Summarizer for a data-science LangGraph workflow.
Use the provided workflow evidence to produce a concise technical report.
Return JSON only. Do not invent artifacts, metrics, or model results.
"""


def build_technical_summary_prompt(
    *,
    summary_context: dict[str, Any],
    fallback: dict[str, Any],
) -> str:
    """Build the final technical summary prompt from bounded workflow evidence."""

    return "\n".join(
        [
            "=== TECHNICAL SUMMARY TASK ===",
            "",
            "Create a final technical report payload from the completed workflow evidence.",
            "This report will be saved as a Markdown file for stakeholder review.",
            "",
            "CRITICAL RULES:",
            "1. Do NOT invent metrics, model names, or artifact paths.",
            "2. Reference ONLY data provided in the workflow_evidence section below.",
            "3. If a field is missing or empty in evidence, note it as '(not available)' in the report.",
            "4. Do NOT make up validation metrics, baseline metrics, or model performance numbers.",
            "5. Organize artifacts logically: group by phase (preliminary_eda, data_cleaning, etc.).",
            "6. report_markdown must be plain Markdown for business readers; do NOT use JSON code blocks.",
            "",
            "Return JSON with these REQUIRED keys:",
            "- executive_summary: string (max 1 sentence, format: '{task_type} workflow for target {target_column} with best model {model_name}')",
            "- limitations: list of strings (derived from problem_understanding.risks and workflow-detected issues)",
            "- report_markdown: string containing complete Markdown report with all sections below",
            "",
            "Required Markdown sections (must be included):",
            "# Technical Report",
            "## Task (use executive_summary, task_summary, task_type, target, candidate_metrics)",
            "## Data Summary (include data_overview: rows, columns, column types in bullet list or table)",
            "## EDA Findings (render as bullet list/table, not JSON)",
            "## Cleaning Summary (render as bullet list/table, not JSON)",
            "## Feature Engineering Summary (selected_features, dropped_features, transform_summary, leakage_checks)",
            "## Model Results (candidate_models, best_model_name, best_params, validation_metrics, baseline_metrics)",
            "## Reviewer History (list each phase's review status and comment)",
            "## Artifacts (list all artifact paths by phase)",
            "## Limitations (converted from the above limitations list)",
            "",
            "Artifact organization guidance:",
            "- Group artifacts by workflow phase: reader, preliminary_eda, data_cleaning, feature_engineering, model_building",
            "- Format: `- Phase: [artifact_key]: [artifact_path]`",
            "- Do NOT list artifacts that are missing or have empty paths",
            "",
            "Workflow evidence (this is your ONLY source of truth):",
            json.dumps(summary_context, ensure_ascii=False, indent=2, default=str),
            "",
            "Deterministic fallback payload (use if evidence is incomplete):",
            json.dumps(fallback, ensure_ascii=False, indent=2, default=str),
        ]
    )


__all__ = ["SUMMARIZER_SYSTEM_PROMPT", "build_technical_summary_prompt"]
