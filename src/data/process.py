"""Data processing pipeline.

Reads raw CSVs, casts types per schema, validates data quality,
and saves to processed Parquet files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schemas import (
    REQUIRED_COLUMNS,
    TEST_SCHEMA,
    TRAIN_SCHEMA,
    VALID_RANGES,
)
from src.data.validate import validate_dataset
from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import write_parquet

logger = setup_logger(__name__)


def _cast_dtypes(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Cast DataFrame columns to expected dtypes."""
    for col, dtype in schema.items():
        if col in df.columns and str(df[col].dtype) != dtype:
            logger.info("casting_dtype", column=col, from_dtype=str(df[col].dtype), to_dtype=dtype)
            df[col] = df[col].astype(dtype)
    return df


def _process_dataset(
    raw_path: Path,
    out_path: Path,
    schema: dict[str, str],
    dataset_name: str,
) -> None:
    """Read, cast, validate, and save a single dataset."""
    logger.info("reading_raw", path=str(raw_path), dataset=dataset_name)
    df = pd.read_csv(raw_path)
    logger.info("raw_shape", rows=len(df), columns=len(df.columns), dataset=dataset_name)

    df = _cast_dtypes(df, schema)

    # Filter valid ranges to columns present in this dataset
    ranges = {k: v for k, v in VALID_RANGES.items() if k in df.columns}
    required = [c for c in REQUIRED_COLUMNS if c in df.columns]

    validate_dataset(
        df,
        schema=schema,
        required_columns=required,
        valid_ranges=ranges,
        expected_rows=len(df),
    )

    write_parquet(df, out_path)
    logger.info("saved_processed", path=str(out_path), dataset=dataset_name)


def process_train() -> None:
    """Process raw training data and save as Parquet."""
    config = load_config(Path("configs/data_config.yaml"))
    raw_path = Path(config["paths"]["raw"]) / config["raw_files"]["train"]
    out_path = Path(config["paths"]["processed"]) / config["processed_files"]["train"]
    _process_dataset(raw_path, out_path, TRAIN_SCHEMA, "train")


def process_test() -> None:
    """Process raw test data and save as Parquet."""
    config = load_config(Path("configs/data_config.yaml"))
    raw_path = Path(config["paths"]["raw"]) / config["raw_files"]["test"]
    out_path = Path(config["paths"]["processed"]) / config["processed_files"]["test"]
    _process_dataset(raw_path, out_path, TEST_SCHEMA, "test")


def main() -> None:
    """Run the full data processing pipeline."""
    logger.info("pipeline_start")
    process_train()
    process_test()
    logger.info("pipeline_complete")


if __name__ == "__main__":
    main()
