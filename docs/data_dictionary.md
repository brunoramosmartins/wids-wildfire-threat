# Data Dictionary

## Overview

- **Source:** WiDS Global Datathon 2026 — Watch Duty wildfire data
- **Train:** 221 rows, 37 columns (35 features + 2 targets)
- **Test:** 95 rows, 35 columns (no target columns)
- **Null values:** Zero across all columns
- **Submission:** `event_id` + `prob_12h`, `prob_24h`, `prob_48h`, `prob_72h`

## Column Definitions

### Identifier

| Column | Type | Description | Unique | Notes |
|--------|------|-------------|--------|-------|
| `event_id` | int64 | Anonymized fire event identifier (stable random remap, no temporal meaning) | 221 | Join key for submission |

### Target

| Column | Type | Description | Range | Notes |
|--------|------|-------------|-------|-------|
| `time_to_hit_hours` | float64 | Time from t0+5h until fire comes within 5km of evac zone. For censored events, this is the last observed time within the 72h window. | [0, 72] | **Survival time** — used for training survival models |
| `event` | int64 | Event indicator: 1 if fire hit within 72h, 0 if censored (never hit) | {0, 1} | **Censoring indicator** — 152 censored (68.8%), 69 hits (31.2%) |

### Temporal Coverage

| Column | Type | Description | Null Rate | Notes |
|--------|------|-------------|-----------|-------|
| `num_perimeters_0_5h` | int64 | Number of perimeters observed within first 5 hours | 0% | Range [1, 17], median=1 |
| `dt_first_last_0_5h` | float64 | Time span between first and last perimeter (hours) | 0% | 0 when only 1 perimeter |
| `low_temporal_resolution_0_5h` | int64 | Flag: 1 if dt < 0.5h or only 1 perimeter, else 0 | 0% | 72.9% are low resolution |

### Growth

| Column | Type | Description | Null Rate | Notes |
|--------|------|-------------|-----------|-------|
| `area_first_ha` | float64 | Initial fire area at t0 (hectares) | 0% | Wide range [0.04, 11942] |
| `area_growth_abs_0_5h` | float64 | Absolute area growth in 0-5h window | 0% | High sparsity — 75% are zero |
| `area_growth_rel_0_5h` | float64 | Relative area growth in 0-5h window | 0% | High sparsity |
| `area_growth_rate_ha_per_h` | float64 | Area growth rate (hectares per hour) | 0% | High sparsity |
| `log1p_area_first` | float64 | log(1 + area_first_ha) | 0% | Derived from `area_first_ha` |
| `log1p_growth` | float64 | log(1 + growth) | 0% | Derived from growth |
| `log_area_ratio_0_5h` | float64 | Log area ratio in 0-5h window | 0% | Derived |
| `relative_growth_0_5h` | float64 | Relative growth in 0-5h window | 0% | Same as `area_growth_rel_0_5h` — **highly correlated (r~1.0)** |
| `radial_growth_m` | float64 | Radial growth in meters | 0% | High sparsity |
| `radial_growth_rate_m_per_h` | float64 | Radial growth rate (meters per hour) | 0% | High sparsity |

### Centroid Kinematics

| Column | Type | Description | Null Rate | Notes |
|--------|------|-------------|-----------|-------|
| `centroid_displacement_m` | float64 | Centroid displacement in meters | 0% | High sparsity (75% zero) |
| `centroid_speed_m_per_h` | float64 | Centroid movement speed (meters/hour) | 0% | High sparsity |
| `spread_bearing_deg` | float64 | Fire spread direction in degrees | 0% | 0 when no movement |
| `spread_bearing_sin` | float64 | sin(spread_bearing) — directional encoding | 0% | Circular encoding |
| `spread_bearing_cos` | float64 | cos(spread_bearing) — directional encoding | 0% | Circular encoding |

### Distance

| Column | Type | Description | Null Rate | Notes |
|--------|------|-------------|-----------|-------|
| `dist_min_ci_0_5h` | float64 | Minimum distance to nearest evac zone centroid (meters) | 0% | **Key predictor** — range [307, 757700] |
| `dist_std_ci_0_5h` | float64 | Std deviation of distances in 0-5h window | 0% | High sparsity |
| `dist_change_ci_0_5h` | float64 | Change in distance during 0-5h window | 0% | Positive = moving away |
| `dist_slope_ci_0_5h` | float64 | Slope of distance over time | 0% | Trend direction |
| `closing_speed_m_per_h` | float64 | Speed at which fire closes distance to evac zone (m/h, positive=closing) | 0% | **Key predictor** |
| `closing_speed_abs_m_per_h` | float64 | Absolute closing speed | 0% | Highly correlated with `closing_speed_m_per_h` |
| `projected_advance_m` | float64 | Projected advance toward evac zone | 0% | Derived from closing speed |
| `dist_accel_m_per_h2` | float64 | Acceleration of distance change | 0% | Second derivative |
| `dist_fit_r2_0_5h` | float64 | R-squared of distance linear fit | 0% | Fit quality |

### Directionality

| Column | Type | Description | Null Rate | Notes |
|--------|------|-------------|-----------|-------|
| `alignment_cos` | float64 | Cosine alignment between fire motion and evac direction | 0% | [-1, 1], negative = moving toward |
| `alignment_abs` | float64 | Absolute alignment (0-1, higher = more aligned) | 0% | **Key predictor** |
| `cross_track_component` | float64 | Cross-track component of fire motion | 0% | Lateral movement |
| `along_track_speed` | float64 | Along-track speed toward evac zone | 0% | Longitudinal movement |

### Temporal Metadata

| Column | Type | Description | Null Rate | Notes |
|--------|------|-------------|-----------|-------|
| `event_start_hour` | int64 | Hour of day when fire started (0-23) | 0% | Discrete |
| `event_start_dayofweek` | int64 | Day of week (0=Monday, 6=Sunday) | 0% | Discrete |
| `event_start_month` | int64 | Month when fire started (1-12) | 0% | Range [1, 9] in train |

## Special Handling Notes

### High Sparsity
The following features have >50% zero values and represent fires with no observed movement in the 0-5h window:
- All growth features except `area_first_ha` and `log1p_area_first`
- `centroid_displacement_m`, `centroid_speed_m_per_h`
- `spread_bearing_deg/sin/cos` (partially)
- `dist_std_ci_0_5h`, `dist_change_ci_0_5h`

### Highly Correlated Pairs (|r| > 0.9)
- `area_growth_rel_0_5h` ~ `relative_growth_0_5h` (redundant)
- `closing_speed_m_per_h` ~ `closing_speed_abs_m_per_h`
- `radial_growth_m` ~ `radial_growth_rate_m_per_h`
- Multiple growth-related features are highly correlated

### Leakage Risk
- **None identified** — all features are computed from the 0-5h observation window (before the prediction horizon starts)
- Target `time_to_hit_hours` is the time **after** t0+5h, so features use only pre-horizon data
