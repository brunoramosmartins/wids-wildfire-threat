"""Data schema definitions.

Expected columns, dtypes, required fields, and valid ranges
for train and test datasets.
"""

from __future__ import annotations

# Column name -> expected dtype (TBD after data inspection)
TRAIN_SCHEMA: dict[str, str] = {}
TEST_SCHEMA: dict[str, str] = {}

# Column lists
REQUIRED_COLUMNS: list[str] = []
NUMERIC_COLUMNS: list[str] = []
CATEGORICAL_COLUMNS: list[str] = []

# Column name -> (min, max) valid range
VALID_RANGES: dict[str, tuple[float, float]] = {}
