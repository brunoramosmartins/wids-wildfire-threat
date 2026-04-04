"""Geospatial feature functions.

Distance-based derived features, threat scores, interaction features,
and growth interactions from pre-computed distance/directionality columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distance, threat, and growth interaction features."""
    out = pd.DataFrame(index=df.index)

    dist = df["dist_min_ci_0_5h"]
    speed = df["closing_speed_m_per_h"]
    growth_rate = df["area_growth_rate_ha_per_h"]

    # Distance & threat
    out["threat_score"] = speed / (dist + 1)
    out["projected_arrival_h"] = np.clip(dist / np.maximum(speed, 1), 0, 200)
    out["is_closing"] = (speed > 0).astype(np.int64)
    out["dist_min_log"] = np.log1p(dist)
    out["dist_bin"] = pd.qcut(dist, q=3, labels=False, duplicates="drop").astype(np.int64)
    out["advance_ratio"] = df["projected_advance_m"] / (dist + 1)
    out["alignment_x_speed"] = df["alignment_abs"] * speed

    # Growth interactions
    out["is_growing"] = (df["area_growth_abs_0_5h"] > 0).astype(np.int64)
    out["growth_x_proximity"] = growth_rate / (dist + 1)
    out["speed_x_growth"] = speed * growth_rate

    return out
