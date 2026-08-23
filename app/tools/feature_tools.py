import numpy as np
import pandas as pd
from typing import List, Dict


def extract_regex(
    df: pd.DataFrame, column: str, pattern: str, new_col: str
) -> pd.DataFrame:
    """
    使用正则表达式提取文本。例如从 Name 中提取 Title。
    pattern: 建议带一个 capture group ()。如果意外给了多个 capture groups，
    为了保持单列输出语义，自动取第一个分组结果写入 new_col。
    """
    if column in df.columns:
        extracted = df[column].str.extract(pattern, expand=False)
        if isinstance(extracted, pd.DataFrame):
            if extracted.shape[1] == 0:
                return df
            extracted = extracted.iloc[:, 0]
        df[new_col] = extracted
    return df


def create_binning(
    df: pd.DataFrame,
    column: str,
    n_bins: int,
    new_col: str,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """
    对连续变量进行分箱。strategy="quantile" 使用 qcut, strategy="uniform" 使用 cut。
    非数值列静默跳过。
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df

    try:
        if strategy == "quantile":
            df[new_col] = pd.qcut(df[column], q=n_bins, labels=False, duplicates="drop")
        elif strategy == "uniform":
            df[new_col] = pd.cut(df[column], bins=n_bins, labels=False)
        else:
            raise ValueError("strategy must be 'quantile' or 'uniform'")
    except Exception:
        # 如果数据全是同一个值等难以分箱的异常情况，回退不进行分箱
        df[new_col] = 0
    return df


def create_interaction(
    df: pd.DataFrame, cols: List[str], operator: str, new_col: str
) -> pd.DataFrame:
    """
    两列或多列数值特征交互。支持 +, *, -, /
    自动跳过非数值列参与的交互。
    """
    if not set(cols).issubset(df.columns):
        return df

    # 检查所有参与列必须是数值型
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return df

    if operator == "+":
        df[new_col] = df[cols].sum(axis=1)
    elif operator == "*":
        df[new_col] = df[cols].prod(axis=1)
    # 对于 -, /, 仅实现双列的情况
    elif operator == "-" and len(cols) == 2:
        df[new_col] = df[cols[0]] - df[cols[1]]
    elif operator == "/" and len(cols) == 2:
        # 避免除以 0
        df[new_col] = df[cols[0]] / (df[cols[1]].replace(0, np.nan))
    else:
        raise ValueError("Unsupported interaction operator or invalid column count.")

    return df


def group_rare_categories(
    df: pd.DataFrame,
    column: str,
    threshold: float = 0.05,
    rare_label: str = "Other",
) -> pd.DataFrame:
    """
    将低频类别归为 rare_label。threshold 是最低占比阈值。
    """
    if column not in df.columns:
        return df

    freqs = df[column].value_counts(normalize=True)
    rare_cats = freqs[freqs < threshold].index.tolist()

    if rare_cats:
        df[column] = df[column].where(~df[column].isin(rare_cats), other=rare_label)
    return df


def encode_categorical(
    df: pd.DataFrame,
    column: str,
    method: str,
    mapping: Dict[str, int] = None,
) -> pd.DataFrame:
    """
    将类别列编码。
    method 支持:
      - "ordinal"：必须提供 mapping 字典
      - "label"：自动分配整型编码
      - "onehot"：使用 get_dummies 展开，自动转换为 int8 节省内存
    """
    if column not in df.columns:
        return df

    if method == "ordinal":
        if not mapping:
            raise ValueError(
                "mapping dictionary must be provided for ordinal encoding."
            )
        df[column] = df[column].map(mapping)
    elif method == "label":
        # 基础标签编码
        df[column] = df[column].astype("category").cat.codes
    elif method == "onehot":
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
        # 确保 dummy 列为数值型 (int8)，避免 bool dtype 导致下游问题
        for col in dummies.columns:
            dummies[col] = dummies[col].astype("int8")
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return df
