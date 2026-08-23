import numpy as np
import pandas as pd
from typing import List, Union


def drop_columns(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    删除指定的列。
    """
    if isinstance(columns, str):
        columns = [columns]

    # 只删除真实存在的列
    cols_to_drop = [col for col in columns if col in df.columns]
    return df.drop(columns=cols_to_drop)


def fill_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    fill_value: any = None,
    group_cols: List[str] = None,
) -> pd.DataFrame:
    """
    填充缺失值。支持 constant, mean, median, mode 等策略。
    可以通过 group_cols 实现按特定组别进行填充。

    对于字符串/类别列，mean/median 策略自动降级为 mode。
    """
    if column not in df.columns:
        return df

    series = df[column]
    is_string = pd.api.types.is_string_dtype(series) or series.dtype == "object"

    if strategy == "constant":
        if fill_value is None:
            raise ValueError("strategy='constant' requires 'fill_value' to be provided.")
        df[column] = series.fillna(fill_value)

    elif strategy in ["mean", "median", "mode"]:
        # 字符串/类别列不支持 mean/median，自动降级为 mode
        effective_strategy = strategy
        if is_string and strategy in ["mean", "median"]:
            effective_strategy = "mode"

        if group_cols and set(group_cols).issubset(df.columns):
            # 按照 group 分组填充
            if effective_strategy == "mean":
                df[column] = df.groupby(group_cols)[column].transform(
                    lambda x: x.fillna(x.mean())
                )
            elif effective_strategy == "median":
                df[column] = df.groupby(group_cols)[column].transform(
                    lambda x: x.fillna(x.median())
                )
            elif effective_strategy == "mode":
                df[column] = df.groupby(group_cols)[column].transform(
                    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x)
                )
        else:
            # 全局填充
            if effective_strategy == "mean":
                val = series.mean()
            elif effective_strategy == "median":
                val = series.median()
            elif effective_strategy == "mode":
                mode_result = series.mode()
                val = mode_result.iloc[0] if not mode_result.empty else None
            if val is not None:
                df[column] = series.fillna(val)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


def clip_outliers(
    df: pd.DataFrame,
    column: str,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    对指定的连续数值列应用盖帽法截断异常值。
    非数值列静默跳过。
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df

    lower_bound = df[column].quantile(lower_quantile)
    upper_bound = df[column].quantile(upper_quantile)

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df
