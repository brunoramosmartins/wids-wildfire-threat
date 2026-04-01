"""Weather and environment feature functions.

Wind speed, direction, temperature, humidity aggregations,
and fire weather index approximations.
"""

from __future__ import annotations

import pandas as pd


def compute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all weather features."""
    raise NotImplementedError
