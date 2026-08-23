import warnings

import numpy as np
import pandas as pd


def profile_dataset(
    df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    target_col: str | None = None,
) -> dict:
    """Generate a statistical dataset profile for use as LLM Planner context.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze, typically the training set.
    target_cols : list[str] | None
        Target column list. Supports single-target, multi-target, and multilabel tasks.
    target_col : str | None
        **Deprecated** legacy single-target argument kept for backward compatibility.
        Ignored when ``target_cols`` is also provided.
    """
    # Backward compatibility: legacy signature target_col -> target_cols
    if target_cols is None and target_col is not None:
        warnings.warn(
            "profile_dataset(target_col=...) is deprecated, use target_cols=[...] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        target_cols = [target_col]

    # Normalize inputs
    if target_cols is None:
        target_cols = []
    target_cols = [c for c in target_cols if c in df.columns]

    profile: dict = {
        "dataset_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "target_variable": target_cols if target_cols else None,
        "columns": {},
    }

    # Target profile
    if target_cols:
        for tc in target_cols:
            series = df[tc]
            if _is_classification_target(series):
                # Classification target: class distribution
                profile.setdefault("class_distribution", {})
                profile["class_distribution"][tc] = (
                    series.value_counts(normalize=True).to_dict()
                )
            else:
                # Regression target: continuous-value summary statistics
                profile.setdefault("target_statistics", {})
                profile["target_statistics"][tc] = {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        # Multilabel co-occurrence matrix, generated when there are at least two binary target columns
        binary_targets = [
            tc for tc in target_cols
            if set(df[tc].dropna().unique()).issubset({0, 1, 0.0, 1.0, True, False})
        ]
        if len(binary_targets) >= 2:
            corr_matrix = df[binary_targets].corr(method="pearson")
            # Keep only the upper triangle to reduce JSON size
            cooccurrence = {}
            for i, col_a in enumerate(binary_targets):
                for col_b in binary_targets[i + 1:]:
                    cooccurrence[f"{col_a}__x__{col_b}"] = round(
                        float(corr_matrix.loc[col_a, col_b]), 4
                    )
            profile["label_cooccurrence"] = cooccurrence

    # Column-level profile
    for col in df.columns:
        series = df[col]
        missing_rate = series.isnull().mean()
        dtype = str(series.dtype)

        col_profile: dict = {
            "dtype": dtype,
            "missing_rate": float(missing_rate),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_profile["type"] = "numeric"
            if len(series.dropna()) > 0:
                col_profile["mean"] = float(series.mean())
                col_profile["min"] = float(series.min())
                col_profile["max"] = float(series.max())
                col_profile["nunique"] = int(series.nunique())
        else:
            col_profile["type"] = "categorical"
            col_profile["nunique"] = int(series.nunique())
            if len(series.dropna()) > 0:
                top_counts = series.value_counts(normalize=True).head(3).to_dict()
                col_profile["top_values"] = top_counts

        profile["columns"][col] = col_profile

    return profile


def _is_classification_target(series: pd.Series) -> bool:
    """Determine whether a target column should be treated as classification or regression.

    Rule: string, object, and bool columns are classification; numeric columns
    with ``nunique <= 20`` are also classification; otherwise treat as regression.
    """
    if pd.api.types.is_string_dtype(series) or series.dtype == "bool":
        return True
    if pd.api.types.is_numeric_dtype(series) and series.nunique() <= 20:
        return True
    return False
