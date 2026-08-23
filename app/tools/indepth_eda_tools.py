import json
import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def target_analysis(df: pd.DataFrame, target: str|list[str], model_type: str) -> dict:
    
    # Normalize target to a list
    targets = [target] if isinstance(target, str) else target

    target_analysis = {}
    for target in targets:
        if target in df.columns:
            y = df[target]
            # Initialize section
            target_analysis[target] = {
            "variable": target,
            "model_type": model_type,
            "summary": y.describe().to_dict(),
            }
            
            if re.search(r"classification", str(model_type), re.IGNORECASE):
                target_analysis[target]["value_counts"] = y.value_counts().to_dict()
    return target_analysis

def feature_feature_relationships(
    df: pd.DataFrame,
    target: str | list[str],
    exclude_columns: list[str] | None = None,
) -> dict:
    feature_feature_relationships = {}
    excluded = set(exclude_columns or [])
    targets = [target] if isinstance(target, str) else list(target or [])
    excluded.update(targets)
    num_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in excluded
    ]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        feature_feature_relationships = corr.to_dict()
    else:
        feature_feature_relationships = {
        "note": "Need at least 2 numeric columns for correlation matrix.",
        "numeric_columns": num_cols,
    }
    return feature_feature_relationships

def outlier_analysis(
    df: pd.DataFrame,
    target: str | list[str],
    exclude_columns: list[str] | None = None,
) -> dict:
    outlier_analysis = {}
    excluded = set(exclude_columns or [])
    targets = [target] if isinstance(target, str) else list(target or [])
    excluded.update(targets)
    num_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in excluded
    ]
    for col in num_cols:
        outlier_analysis[col] = {}
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_analysis[col]["lower_bound"] = lower_bound
        outlier_analysis[col]["upper_bound"] = upper_bound
        outlier_analysis[col]["outlier_count"] = len(outliers)
        outlier_analysis[col]["outlier_rate"] = len(outliers) / len(df)
    return outlier_analysis

_MAX_SEGMENT_UNIQUE = 20   # cap per segment column to avoid cartesian product explosion
_MAX_SEGMENT_PRODUCT = 10_000  # max total groups before skipping


def segment_analysis(df: pd.DataFrame, targets: str|list[str], segments: str|list[str], model_type: str) -> dict:
    targets = [targets] if isinstance(targets, str) else targets
    segments = [segments] if isinstance(segments, str) else segments
    segment_analysis = {}

    def __convert_to_bin(series: pd.Series) -> pd.Series:
        # Bin continuous columns (float or int with many unique values) to ≤ 4 categories.
        if pd.api.types.is_float_dtype(series) or (
            pd.api.types.is_integer_dtype(series) and series.nunique() > _MAX_SEGMENT_UNIQUE
        ):
            try:
                return pd.qcut(series, q=4, duplicates="drop")
            except Exception:
                try:
                    return pd.cut(series, bins=4)
                except Exception:
                    return series
        return series

    if not all(f in df.columns for f in segments):
        return {
            "segments": segments,
            "Note": "Not all segments are in the dataframe. No segment analysis performed.",
        }

    df_copy = df.copy()
    for feature in segments:
        df_copy[feature] = __convert_to_bin(df_copy[feature])

    # Guard: skip if the cartesian product of unique values per segment is too large.
    import math
    unique_counts = [df_copy[f].nunique() for f in segments]
    try:
        product = math.prod(unique_counts)
    except Exception:
        product = float("inf")
    if product > _MAX_SEGMENT_PRODUCT:
        return {
            "segments": segments,
            "Note": (
                f"Segment product ({product:,}) exceeds limit ({_MAX_SEGMENT_PRODUCT:,}). "
                "Reduce the number of segment columns or choose lower-cardinality features."
            ),
        }

    for target in targets:
        segment_analysis[target] = {}
        try:
            if re.search(r"classification", str(model_type), re.IGNORECASE):
                key = "|".join(segments) + "_pct"
                prop = (df_copy.groupby(segments)[target]).value_counts(normalize=True)
                segment_analysis[target][key] = {
                    "|".join(map(str, idx if isinstance(idx, tuple) else (idx,))): value
                    for idx, value in prop.to_dict().items()
                }
            elif re.search(r"regression", str(model_type), re.IGNORECASE):
                key = "|".join(segments) + "_mean"
                mean = (df_copy.groupby(segments)[target]).mean()
                segment_analysis[target][key] = {
                    "|".join(map(str, idx if isinstance(idx, tuple) else (idx,))): value
                    for idx, value in mean.to_dict().items()
                }
        except Exception as exc:
            segment_analysis[target]["error"] = str(exc)
    return segment_analysis
