import pytest
import pandas as pd
import numpy as np

from app.tools.profiler import profile_dataset
from app.tools.cleaning_tools import drop_columns, fill_missing, clip_outliers
from app.tools.feature_tools import extract_regex, create_binning, encode_categorical

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [22.0, np.nan, 38.0, 35.0, 1000.0],
        "Pclass": [3, 1, 3, 1, 2],
        "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John", "Heikkinen, Miss. Laina", "Futrelle, Mrs. Jacques Heath", "Allen, Mr. William"],
        "Cabin": ["NaN", "C85", np.nan, "C123", np.nan]
    })

def test_profiler_schema(sample_df):
    profile = profile_dataset(sample_df)
    assert profile["dataset_shape"]["rows"] == 5
    assert profile["dataset_shape"]["columns"] == 4
    assert profile["columns"]["Age"]["type"] == "numeric"
    assert profile["columns"]["Name"]["type"] == "categorical"
    assert profile["columns"]["Age"]["missing_rate"] == 0.2

def test_drop_columns(sample_df):
    df = sample_df.copy()
    result = drop_columns(df, ["Cabin"])
    assert "Cabin" not in result.columns

def test_fill_missing_global(sample_df):
    df = sample_df.copy()
    result = fill_missing(df, "Age", strategy="median")
    assert result["Age"].isnull().sum() == 0
    # Median of [22, 38, 35, 1000] is (35+38)/2 = 36.5
    assert result["Age"].iloc[1] == 36.5

def test_fill_missing_by_group(sample_df):
    df = sample_df.copy()
    # Mocking different pclass values for deterministic median
    df.loc[0, "Age"] = 20
    df.loc[2, "Age"] = 30
    # Pclass 1: [35.0] -> median 35.0
    # Pclass 3: [20, 30] -> median 25.0
    result = fill_missing(df, "Age", strategy="median", group_cols=["Pclass"])
    assert result["Age"].iloc[1] == 35.0  # Pclass 1, missing originally

def test_extract_regex(sample_df):
    df = sample_df.copy()
    result = extract_regex(df, "Name", r'([A-Za-z]+)\.', "Title")
    assert list(result["Title"]) == ["Mr", "Mrs", "Miss", "Mrs", "Mr"]


def test_extract_regex_uses_first_capture_group_when_pattern_has_multiple_groups():
    df = pd.DataFrame({"Cabin": ["B/0/P", "C/123/S", np.nan]})
    result = extract_regex(df, "Cabin", r"([A-Za-z]+)/(\d+)/([A-Za-z]+)", "Cabin_deck")
    assert list(result["Cabin_deck"]) == ["B", "C", np.nan]

def test_encode_categorical(sample_df):
    df = sample_df.copy()
    df["Title"] = ["Mr", "Mrs", "Miss", "Mrs", "Mr"]
    mapping = {"Mr": 0, "Mrs": 1, "Miss": 2}
    result = encode_categorical(df, "Title", method="ordinal", mapping=mapping)
    assert list(result["Title"]) == [0, 1, 2, 1, 0]
