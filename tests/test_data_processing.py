"""Tests for data processing pipeline.

These tests require raw data files in data/raw/ and are skipped in CI
where those files are not available (Kaggle data is not committed).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.process import process_test, process_train
from src.data.schemas import TEST_SCHEMA, TRAIN_SCHEMA

_RAW_DATA_AVAILABLE = Path("data/raw/train.csv").exists()
_skip_no_data = pytest.mark.skipif(
    not _RAW_DATA_AVAILABLE,
    reason="Raw data files not present (Kaggle data not committed to repo)",
)


@_skip_no_data
def test_process_creates_parquet_files() -> None:
    """Pipeline produces both Parquet output files."""
    process_train()
    process_test()
    assert Path("data/processed/train_processed.parquet").exists()
    assert Path("data/processed/test_processed.parquet").exists()


@_skip_no_data
def test_train_schema_matches() -> None:
    """Processed train dtypes match TRAIN_SCHEMA."""
    process_train()
    df = pd.read_parquet("data/processed/train_processed.parquet")
    for col, expected_dtype in TRAIN_SCHEMA.items():
        assert str(df[col].dtype) == expected_dtype, (
            f"{col}: expected {expected_dtype}, got {df[col].dtype}"
        )


@_skip_no_data
def test_test_schema_matches() -> None:
    """Processed test dtypes match TEST_SCHEMA."""
    process_test()
    df = pd.read_parquet("data/processed/test_processed.parquet")
    for col, expected_dtype in TEST_SCHEMA.items():
        assert str(df[col].dtype) == expected_dtype, (
            f"{col}: expected {expected_dtype}, got {df[col].dtype}"
        )


@_skip_no_data
def test_row_counts_preserved() -> None:
    """Processed datasets have same row counts as raw."""
    process_train()
    process_test()
    raw_train = pd.read_csv("data/raw/train.csv")
    raw_test = pd.read_csv("data/raw/test.csv")
    proc_train = pd.read_parquet("data/processed/train_processed.parquet")
    proc_test = pd.read_parquet("data/processed/test_processed.parquet")
    assert len(proc_train) == len(raw_train)
    assert len(proc_test) == len(raw_test)


@_skip_no_data
def test_idempotency() -> None:
    """Running pipeline twice produces identical output."""
    process_train()
    first_run = pd.read_parquet("data/processed/train_processed.parquet")
    process_train()
    second_run = pd.read_parquet("data/processed/train_processed.parquet")
    pd.testing.assert_frame_equal(first_run, second_run)
