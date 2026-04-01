"""Feature selection utilities.

Recursive Feature Elimination (RFE), permutation importance,
and feature set comparison.
"""

from __future__ import annotations

import pandas as pd


def select_features(df: pd.DataFrame, target: pd.Series) -> list[str]:
    """Select top features using importance-based methods."""
    raise NotImplementedError
