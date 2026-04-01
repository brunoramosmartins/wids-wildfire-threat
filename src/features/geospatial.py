"""Geospatial feature functions.

Distance calculations, bearing, region encoding, and spatial density features.
"""

from __future__ import annotations

import pandas as pd


def compute_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all geospatial features."""
    raise NotImplementedError
