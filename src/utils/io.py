"""I/O utilities.

Read and write Parquet, YAML, and other file formats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Write a DataFrame to a Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
