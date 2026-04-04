"""Tests for Kaggle submission format validation."""

import numpy as np
import pandas as pd
import pytest

from src.submission.format import validate_submission


def _make_valid_submission(n: int = 5) -> pd.DataFrame:
    """Create a valid submission DataFrame."""
    return pd.DataFrame(
        {
            "event_id": range(1, n + 1),
            "prob_12h": np.linspace(0.1, 0.3, n),
            "prob_24h": np.linspace(0.2, 0.4, n),
            "prob_48h": np.linspace(0.3, 0.5, n),
            "prob_72h": np.linspace(0.4, 0.6, n),
        }
    )


def test_valid_submission_passes() -> None:
    """A correctly formatted submission passes validation."""
    df = _make_valid_submission()
    validate_submission(df)  # Should not raise


def test_wrong_columns_fails() -> None:
    """Missing or extra columns should fail."""
    df = pd.DataFrame({"event_id": [1], "prob_12h": [0.5]})
    with pytest.raises(ValueError, match="Expected columns"):
        validate_submission(df)


def test_null_values_fail() -> None:
    """Null values in submission should fail."""
    df = _make_valid_submission()
    df.loc[0, "prob_12h"] = np.nan
    with pytest.raises(ValueError, match="null values"):
        validate_submission(df)


def test_out_of_range_fails() -> None:
    """Probabilities outside [0, 1] should fail."""
    df = _make_valid_submission()
    df.loc[0, "prob_12h"] = 1.5
    with pytest.raises(ValueError, match="outside"):
        validate_submission(df)


def test_monotonicity_violation_fails() -> None:
    """P(12h) > P(24h) should fail monotonicity check."""
    df = _make_valid_submission()
    df.loc[0, "prob_12h"] = 0.9  # Higher than prob_24h
    with pytest.raises(ValueError, match="Monotonicity"):
        validate_submission(df)


def test_row_count_mismatch_fails() -> None:
    """Wrong number of rows should fail when expected IDs provided."""
    df = _make_valid_submission(n=5)
    expected_ids = pd.Series(range(1, 11))  # 10 expected, 5 provided
    with pytest.raises(ValueError, match="Expected 10"):
        validate_submission(df, expected_ids=expected_ids)
