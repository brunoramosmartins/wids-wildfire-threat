"""Infrastructure feature functions.

No raw infrastructure data (type, density, vulnerability) is available in
this dataset. All features are pre-computed. This module is a no-op
placeholder that returns an empty DataFrame, preserving the modular
architecture for future extension if external infrastructure data becomes
available.
"""

from __future__ import annotations

import pandas as pd


def compute_infrastructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return empty DataFrame — no raw infrastructure data available."""
    return pd.DataFrame(index=df.index)
