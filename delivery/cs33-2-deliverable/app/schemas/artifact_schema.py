from __future__ import annotations

from pathlib import Path
from typing import Any, Final


DEFAULT_ARTIFACT_PATHS: Final[dict[str, Any]] = {
    "preliminary_eda_report": "artifacts/eda/preliminary_eda.md",
    "cleaned_train_data": "artifacts/data/cleaned_train.csv",
    "cleaned_test_data": "artifacts/data/cleaned_test.csv",
    "cleaning_report": "artifacts/data/cleaning_report.md",
    "indepth_eda_report": "artifacts/eda/in_depth_eda.md",
    "feature_summary": "artifacts/features/feature_summary.json",
    "engineered_train_data": "artifacts/features/engineered_train.csv",
    "engineered_test_data": "artifacts/features/engineered_test.csv",
    "model_results": "artifacts/models/model_results.json",
    "best_model": "artifacts/models/best_model.pkl",
    "prediction_file": "artifacts/predictions/submission.csv",
    "technical_report": "artifacts/reports/technical_report.md",
    "business_report": "artifacts/reports/business_report.md",
    "run_manifest": "artifacts/reports/run_manifest.json",
    "phase_plan_records": {},
    "phase_review_records": {},
    "figures": [],
}

RUN_ARTIFACT_SUBPATHS: Final[dict[str, str]] = {
    "preliminary_eda_report": "eda/preliminary_eda.md",
    "cleaned_train_data": "data/cleaned_train.csv",
    "cleaned_test_data": "data/cleaned_test.csv",
    "cleaning_report": "data/cleaning_report.md",
    "indepth_eda_report": "eda/in_depth_eda.md",
    "feature_summary": "features/feature_summary.json",
    "engineered_train_data": "features/engineered_train.csv",
    "engineered_test_data": "features/engineered_test.csv",
    "model_results": "models/model_results.json",
    "best_model": "models/best_model.pkl",
    "prediction_file": "predictions/submission.csv",
    "technical_report": "reports/technical_report.md",
    "business_report": "reports/business_report.md",
    "run_manifest": "reports/run_manifest.json",
}


def build_run_artifact_root(run_id: str, *, run_label: str | None = None) -> str:
    leaf = str(run_label or run_id).strip()
    return str(Path("artifacts") / "runs" / leaf)


def build_run_artifact_paths(run_root: str) -> dict[str, Any]:
    """Materialize artifact paths under a per-run artifact root."""

    root = Path(run_root)
    return {
        key: str(root / relative_path)
        for key, relative_path in RUN_ARTIFACT_SUBPATHS.items()
    } | {"figures": []}
