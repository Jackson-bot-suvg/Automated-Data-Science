import warnings

import numpy as np
import pandas as pd


def profile_dataset(
    df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    target_col: str | None = None,
) -> dict:
    """生成数据集的统计学画像，供 LLM Planner 作为上下文使用。

    Parameters
    ----------
    df : pd.DataFrame
        要分析的 DataFrame（通常是训练集）。
    target_cols : list[str] | None
        目标列列表。支持单目标、多目标和多标签。
    target_col : str | None
        **Deprecated**. 旧的单目标参数，保留向后兼容。
        如果同时传入 ``target_cols``，则忽略此参数。
    """
    # ── 向后兼容：旧签名 target_col -> target_cols ─────
    if target_cols is None and target_col is not None:
        warnings.warn(
            "profile_dataset(target_col=...) is deprecated, use target_cols=[...] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        target_cols = [target_col]

    # 规范化
    if target_cols is None:
        target_cols = []
    target_cols = [c for c in target_cols if c in df.columns]

    profile: dict = {
        "dataset_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "target_variable": target_cols if target_cols else None,
        "columns": {},
    }

    # ── 目标列画像 ──────────────────────────────────
    if target_cols:
        for tc in target_cols:
            series = df[tc]
            if _is_classification_target(series):
                # 分类目标：类别分布
                profile.setdefault("class_distribution", {})
                profile["class_distribution"][tc] = (
                    series.value_counts(normalize=True).to_dict()
                )
            else:
                # 回归目标：连续值统计
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

        # 多标签共现矩阵（>= 2 个二值目标列时生成）
        binary_targets = [
            tc for tc in target_cols
            if set(df[tc].dropna().unique()).issubset({0, 1, 0.0, 1.0, True, False})
        ]
        if len(binary_targets) >= 2:
            corr_matrix = df[binary_targets].corr(method="pearson")
            # 只保留上三角（减少 JSON 体积）
            cooccurrence = {}
            for i, col_a in enumerate(binary_targets):
                for col_b in binary_targets[i + 1:]:
                    cooccurrence[f"{col_a}__x__{col_b}"] = round(
                        float(corr_matrix.loc[col_a, col_b]), 4
                    )
            profile["label_cooccurrence"] = cooccurrence

    # ── 列级画像 ────────────────────────────────────
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
    """判断目标列是分类还是回归。

    规则：字符串/object/bool 类型 → 分类；nunique <= 20 → 分类；否则 → 回归。
    """
    if pd.api.types.is_string_dtype(series) or series.dtype == "bool":
        return True
    if pd.api.types.is_numeric_dtype(series) and series.nunique() <= 20:
        return True
    return False
