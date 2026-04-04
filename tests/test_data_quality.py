"""Tests for data validation and quality checks."""

import numpy as np
import pandas as pd

from src.data.validate import (
    validate_nulls,
    validate_ranges,
    validate_row_count,
    validate_schema,
)


def _make_df() -> pd.DataFrame:
    """Create a small test DataFrame."""
    return pd.DataFrame(
        {
            "a": pd.array([1, 2, 3], dtype="int64"),
            "b": pd.array([1.0, 2.0, 3.0], dtype="float64"),
        }
    )


# --- validate_schema ---


def test_validate_schema_pass() -> None:
    df = _make_df()
    assert validate_schema(df, {"a": "int64", "b": "float64"}) is True


def test_validate_schema_fail_missing_col() -> None:
    df = _make_df()
    assert validate_schema(df, {"a": "int64", "b": "float64", "c": "int64"}) is False


def test_validate_schema_fail_wrong_dtype() -> None:
    df = _make_df()
    assert validate_schema(df, {"a": "float64", "b": "float64"}) is False


# --- validate_nulls ---


def test_validate_nulls_pass() -> None:
    df = _make_df()
    assert validate_nulls(df, ["a", "b"]) is True


def test_validate_nulls_fail() -> None:
    df = pd.DataFrame({"a": [1, 2, None], "b": [1.0, np.nan, 3.0]})
    assert validate_nulls(df, ["a", "b"]) is False


# --- validate_ranges ---


def test_validate_ranges_pass() -> None:
    df = _make_df()
    assert validate_ranges(df, {"a": (0, 10), "b": (0.0, 5.0)}) is True


def test_validate_ranges_fail() -> None:
    df = _make_df()
    assert validate_ranges(df, {"a": (2, 10)}) is False  # value 1 is below 2


# --- validate_row_count ---


def test_validate_row_count_pass() -> None:
    df = _make_df()
    assert validate_row_count(df, 3) is True


def test_validate_row_count_fail() -> None:
    df = _make_df()
    assert validate_row_count(df, 100) is False
