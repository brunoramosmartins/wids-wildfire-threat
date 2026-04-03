"""Data validation and quality checks.

Validates processed data against expected schemas, null constraints,
and value ranges.
"""

from __future__ import annotations

import pandas as pd

from src.observability.logger import setup_logger

logger = setup_logger(__name__)


def validate_schema(df: pd.DataFrame, expected_dtypes: dict[str, str]) -> bool:
    """Check that DataFrame columns match expected dtypes."""
    valid = True

    missing = set(expected_dtypes) - set(df.columns)
    if missing:
        logger.error("missing_columns", columns=sorted(missing))
        valid = False

    extra = set(df.columns) - set(expected_dtypes)
    if extra:
        logger.warning("extra_columns", columns=sorted(extra))

    for col, expected in expected_dtypes.items():
        if col in df.columns and str(df[col].dtype) != expected:
            logger.error(
                "dtype_mismatch",
                column=col,
                expected=expected,
                actual=str(df[col].dtype),
            )
            valid = False

    return valid


def validate_nulls(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """Check that required columns have no null values."""
    valid = True
    for col in required_columns:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.error("nulls_found", column=col, null_count=int(null_count))
            valid = False

    return valid


def validate_ranges(
    df: pd.DataFrame, valid_ranges: dict[str, tuple[float, float]]
) -> bool:
    """Check that numeric columns are within expected bounds."""
    valid = True
    for col, (low, high) in valid_ranges.items():
        if col not in df.columns:
            continue
        below = (df[col] < low).sum()
        above = (df[col] > high).sum()
        if below > 0 or above > 0:
            logger.error(
                "range_violation",
                column=col,
                expected_min=low,
                expected_max=high,
                below_count=int(below),
                above_count=int(above),
            )
            valid = False

    return valid


def validate_row_count(
    df: pd.DataFrame, expected_count: int, tolerance: float = 0.01
) -> bool:
    """Check that row count is within tolerance of expected count."""
    actual = len(df)
    delta = abs(actual - expected_count) / expected_count if expected_count > 0 else 0
    if delta > tolerance:
        logger.error(
            "row_count_mismatch",
            expected=expected_count,
            actual=actual,
            delta_pct=f"{delta:.2%}",
        )
        return False
    return True


def validate_dataset(
    df: pd.DataFrame,
    schema: dict[str, str],
    required_columns: list[str],
    valid_ranges: dict[str, tuple[float, float]],
    expected_rows: int,
) -> None:
    """Run all validations and raise ValueError if any fail."""
    results = {
        "schema": validate_schema(df, schema),
        "nulls": validate_nulls(df, required_columns),
        "ranges": validate_ranges(df, valid_ranges),
        "row_count": validate_row_count(df, expected_rows),
    }

    failed = [name for name, passed in results.items() if not passed]
    if failed:
        raise ValueError(f"Validation failed: {', '.join(failed)}")

    logger.info("validation_passed", rows=len(df), columns=len(df.columns))
