import numpy as np
import pandas as pd
from typing import List, Dict


def extract_regex(
    df: pd.DataFrame, column: str, pattern: str, new_col: str
) -> pd.DataFrame:
    """
    Extract text using a regular expression, for example extracting Title from Name.
    ``pattern`` is expected to include one capture group. If multiple capture
    groups are provided, the first one is used to preserve single-column output semantics.
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
    Bin a continuous variable.
    ``strategy="quantile"`` uses ``qcut`` and ``strategy="uniform"`` uses ``cut``.
    Non-numeric columns are skipped silently.
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
        # If the data cannot be binned, such as when all values are identical, fall back to a constant bucket
        df[new_col] = 0
    return df


def create_interaction(
    df: pd.DataFrame, cols: List[str], operator: str, new_col: str
) -> pd.DataFrame:
    """
    Create interactions among two or more numeric feature columns.
    Supports ``+``, ``*``, ``-``, and ``/``.
    Non-numeric participating columns are skipped automatically.
    """
    if not set(cols).issubset(df.columns):
        return df

    # All participating columns must be numeric
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return df

    if operator == "+":
        df[new_col] = df[cols].sum(axis=1)
    elif operator == "*":
        df[new_col] = df[cols].prod(axis=1)
    # For - and /, only the two-column case is supported
    elif operator == "-" and len(cols) == 2:
        df[new_col] = df[cols[0]] - df[cols[1]]
    elif operator == "/" and len(cols) == 2:
        # Avoid division by zero
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
    Group infrequent categories into ``rare_label``.
    ``threshold`` is the minimum share cutoff.
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
    Encode a categorical column.
    Supported methods:
      - ``ordinal``: requires a mapping dictionary
      - ``label``: automatically assigns integer codes
      - ``onehot``: expands with ``get_dummies`` and converts outputs to ``int8`` to save memory
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
        # Basic label encoding
        df[column] = df[column].astype("category").cat.codes
    elif method == "onehot":
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
        # Ensure dummy columns are numeric (int8) to avoid downstream bool dtype issues
        for col in dummies.columns:
            dummies[col] = dummies[col].astype("int8")
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return df
