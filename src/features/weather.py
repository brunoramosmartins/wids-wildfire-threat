"""Weather and environment feature functions.

No raw weather data (wind, temperature, humidity) is available in this
dataset. All features are pre-computed. This module is a no-op placeholder
that returns an empty DataFrame, preserving the modular architecture for
future extension if external weather data becomes available.
"""

from __future__ import annotations

import pandas as pd


def compute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return empty DataFrame — no raw weather data available."""
    return pd.DataFrame(index=df.index)
