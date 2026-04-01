"""Infrastructure feature functions.

Infrastructure type encoding, density metrics, and vulnerability scores.
"""

from __future__ import annotations

import pandas as pd


def compute_infrastructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all infrastructure features."""
    raise NotImplementedError
