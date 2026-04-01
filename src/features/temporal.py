"""Temporal and survival feature functions.

Time-of-day, duration since fire start, rolling statistics,
and survival-specific hazard proxy features.
"""

from __future__ import annotations

import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all temporal features."""
    raise NotImplementedError
