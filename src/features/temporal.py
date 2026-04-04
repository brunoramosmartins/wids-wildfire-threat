"""Temporal feature functions.

Cyclical encoding of time variables and time-based binary flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cyclical time encodings and temporal flags."""
    out = pd.DataFrame(index=df.index)

    hour = df["event_start_hour"]
    month = df["event_start_month"]
    dow = df["event_start_dayofweek"]

    # Cyclical encoding
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Binary flags
    out["is_daytime"] = ((hour >= 6) & (hour < 20)).astype(np.int64)
    out["is_weekend"] = (dow >= 5).astype(np.int64)

    return out
