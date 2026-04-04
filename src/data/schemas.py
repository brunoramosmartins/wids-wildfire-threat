"""Data schema definitions.

Expected columns, dtypes, required fields, and valid ranges
for train and test datasets.
"""

from __future__ import annotations

# --- Shared feature columns (present in both train and test) ---

_FEATURE_SCHEMA: dict[str, str] = {
    "event_id": "int64",
    # Temporal coverage
    "num_perimeters_0_5h": "int64",
    "dt_first_last_0_5h": "float64",
    "low_temporal_resolution_0_5h": "int64",
    # Growth
    "area_first_ha": "float64",
    "area_growth_abs_0_5h": "float64",
    "area_growth_rel_0_5h": "float64",
    "area_growth_rate_ha_per_h": "float64",
    "log1p_area_first": "float64",
    "log1p_growth": "float64",
    "log_area_ratio_0_5h": "float64",
    "relative_growth_0_5h": "float64",
    "radial_growth_m": "float64",
    "radial_growth_rate_m_per_h": "float64",
    # Centroid kinematics
    "centroid_displacement_m": "float64",
    "centroid_speed_m_per_h": "float64",
    "spread_bearing_deg": "float64",
    "spread_bearing_sin": "float64",
    "spread_bearing_cos": "float64",
    # Distance
    "dist_min_ci_0_5h": "float64",
    "dist_std_ci_0_5h": "float64",
    "dist_change_ci_0_5h": "float64",
    "dist_slope_ci_0_5h": "float64",
    "closing_speed_m_per_h": "float64",
    "closing_speed_abs_m_per_h": "float64",
    "projected_advance_m": "float64",
    "dist_accel_m_per_h2": "float64",
    "dist_fit_r2_0_5h": "float64",
    # Directionality
    "alignment_cos": "float64",
    "alignment_abs": "float64",
    "cross_track_component": "float64",
    "along_track_speed": "float64",
    # Temporal metadata
    "event_start_hour": "int64",
    "event_start_dayofweek": "int64",
    "event_start_month": "int64",
}

# Column name -> expected dtype
TRAIN_SCHEMA: dict[str, str] = {
    **_FEATURE_SCHEMA,
    "time_to_hit_hours": "float64",
    "event": "int64",
}

TEST_SCHEMA: dict[str, str] = {**_FEATURE_SCHEMA}

# Column lists
REQUIRED_COLUMNS: list[str] = list(_FEATURE_SCHEMA.keys())

NUMERIC_COLUMNS: list[str] = [col for col, dtype in _FEATURE_SCHEMA.items() if col != "event_id"]

CATEGORICAL_COLUMNS: list[str] = []

# Column name -> (min, max) valid range
VALID_RANGES: dict[str, tuple[float, float]] = {
    "event": (0, 1),
    "low_temporal_resolution_0_5h": (0, 1),
    "event_start_hour": (0, 23),
    "event_start_dayofweek": (0, 6),
    "event_start_month": (1, 12),
    "time_to_hit_hours": (0, 72),
    "alignment_cos": (-1, 1),
    "alignment_abs": (0, 1),
    "dist_fit_r2_0_5h": (0, 1),
}
