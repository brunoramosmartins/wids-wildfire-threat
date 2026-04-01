"""Data processing pipeline.

Reads raw CSVs, cleans data, casts types, handles nulls and outliers,
and saves to processed Parquet files.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def process_train() -> None:
    """Process raw training data and save as Parquet."""
    raise NotImplementedError


def process_test() -> None:
    """Process raw test data and save as Parquet."""
    raise NotImplementedError


def main() -> None:
    """Run the full data processing pipeline."""
    logger.info("Starting data processing pipeline")
    process_train()
    process_test()
    logger.info("Data processing complete")


if __name__ == "__main__":
    main()
