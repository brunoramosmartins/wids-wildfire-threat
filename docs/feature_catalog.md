# Feature Catalog

## Overview

All features used in the WiDS Wildfire Time-to-Threat prediction pipeline. Features are organized into **base features** (from processed data) and **engineered features** (derived in Phase 3).

Anti-leakage discipline: every feature uses only data from the 0-5h observation window (before prediction horizon).

## Base Features (34 columns from processed data)

These pass through unchanged from `data/processed/` to `data/features/`.

### Temporal Coverage

| Feature | Type | Source | Rationale | Leakage | Priority |
|---------|------|--------|-----------|---------|----------|
| `num_perimeters_0_5h` | int64 | raw | Observation density; more perimeters = better signal | Safe | Medium |
| `dt_first_last_0_5h` | float64 | raw | Temporal span of observations | Safe | Medium |
| `low_temporal_resolution_0_5h` | int64 | raw | Flag for sparse observation (72.9% of data) | Safe | Medium |

### Growth

| Feature | Type | Source | Rationale | Leakage | Priority |
|---------|------|--------|-----------|---------|----------|
| `area_first_ha` | float64 | raw | Initial fire size -- larger fires more threatening | Safe | High |
| `area_growth_abs_0_5h` | float64 | raw | Absolute area growth in window | Safe | Medium |
| `area_growth_rel_0_5h` | float64 | raw | Relative area growth | Safe | Low (redundant) |
| `area_growth_rate_ha_per_h` | float64 | raw | Growth rate (ha/h) | Safe | Medium |
| `log1p_area_first` | float64 | raw | Log-transformed initial area | Safe | High |
| `log1p_growth` | float64 | raw | Log-transformed growth | Safe | Medium |
| `log_area_ratio_0_5h` | float64 | raw | Log area ratio | Safe | Medium |
| `relative_growth_0_5h` | float64 | raw | Same as area_growth_rel -- drop candidate | Safe | Drop |
| `radial_growth_m` | float64 | raw | Radial growth in meters | Safe | Medium |
| `radial_growth_rate_m_per_h` | float64 | raw | Radial growth rate -- correlated with radial_growth_m | Safe | Low (redundant) |

### Centroid Kinematics

| Feature | Type | Source | Rationale | Leakage | Priority |
|---------|------|--------|-----------|---------|----------|
| `centroid_displacement_m` | float64 | raw | How far fire centroid moved | Safe | Medium |
| `centroid_speed_m_per_h` | float64 | raw | Centroid movement speed | Safe | Medium |
| `spread_bearing_deg` | float64 | raw | Raw bearing -- use sin/cos instead | Safe | Low |
| `spread_bearing_sin` | float64 | raw | Circular encoding of spread direction | Safe | Medium |
| `spread_bearing_cos` | float64 | raw | Circular encoding of spread direction | Safe | Medium |

### Distance

| Feature | Type | Source | Rationale | Leakage | Priority |
|---------|------|--------|-----------|---------|----------|
| `dist_min_ci_0_5h` | float64 | raw | Minimum distance to evac zone -- **strongest predictor** | Safe | Critical |
| `dist_std_ci_0_5h` | float64 | raw | Distance variability in window | Safe | Medium |
| `dist_change_ci_0_5h` | float64 | raw | Distance change (positive = moving away) | Safe | High |
| `dist_slope_ci_0_5h` | float64 | raw | Slope of distance over time | Safe | High |
| `closing_speed_m_per_h` | float64 | raw | Speed closing on evac zone -- **key predictor** | Safe | Critical |
| `closing_speed_abs_m_per_h` | float64 | raw | Absolute closing speed -- correlated with signed | Safe | Low (redundant) |
| `projected_advance_m` | float64 | raw | Projected advance toward evac zone | Safe | High |
| `dist_accel_m_per_h2` | float64 | raw | Acceleration of distance change | Safe | Medium |
| `dist_fit_r2_0_5h` | float64 | raw | R-squared of distance linear fit | Safe | Medium |

### Directionality

| Feature | Type | Source | Rationale | Leakage | Priority |
|---------|------|--------|-----------|---------|----------|
| `alignment_cos` | float64 | raw | Cosine alignment fire to evac | Safe | High |
| `alignment_abs` | float64 | raw | Absolute alignment -- **key predictor** | Safe | Critical |
| `cross_track_component` | float64 | raw | Lateral movement component | Safe | Medium |
| `along_track_speed` | float64 | raw | Longitudinal speed toward evac | Safe | High |

### Temporal Metadata

| Feature | Type | Source | Rationale | Leakage | Priority |
|---------|------|--------|-----------|---------|----------|
| `event_start_hour` | int64 | raw | Hour of day -- fire behavior varies diurnally | Safe | Medium |
| `event_start_dayofweek` | int64 | raw | Day of week | Safe | Low |
| `event_start_month` | int64 | raw | Month -- seasonality of fire behavior | Safe | Medium |

## Engineered Features (derived in Phase 3)

### Distance and Threat (geospatial.py)

| Feature | Type | Computation | Rationale | Leakage | Priority |
|---------|------|-------------|-----------|---------|----------|
| `threat_score` | float64 | `closing_speed / (dist_min + 1)` | Normalized approach rate -- high when fast and close | Safe | Critical |
| `projected_arrival_h` | float64 | `dist_min / max(closing_speed, 1)`, clipped [0, 200] | Estimated hours until arrival | Safe | High |
| `is_closing` | int64 | `1 if closing_speed > 0 else 0` | Binary flag: fire approaching evac zone | Safe | High |
| `dist_min_log` | float64 | `log1p(dist_min)` | Log-scaled distance for better separation | Safe | High |
| `dist_bin` | int64 | Quantile-binned distance (3 bins: near/medium/far) | Categorical distance grouping | Safe | Medium |
| `advance_ratio` | float64 | `projected_advance / (dist_min + 1)` | Fraction of distance covered by projected advance | Safe | High |
| `alignment_x_speed` | float64 | `alignment_abs * closing_speed` | Interaction: aligned AND fast = high threat | Safe | High |

### Temporal (temporal.py)

| Feature | Type | Computation | Rationale | Leakage | Priority |
|---------|------|-------------|-----------|---------|----------|
| `hour_sin` | float64 | `sin(2pi * hour / 24)` | Cyclical encoding of hour | Safe | Medium |
| `hour_cos` | float64 | `cos(2pi * hour / 24)` | Cyclical encoding of hour | Safe | Medium |
| `month_sin` | float64 | `sin(2pi * month / 12)` | Cyclical encoding of month | Safe | Medium |
| `month_cos` | float64 | `cos(2pi * month / 12)` | Cyclical encoding of month | Safe | Medium |
| `is_daytime` | int64 | `1 if 6 <= hour < 20` | Daytime fires behave differently (convection) | Safe | Medium |
| `is_weekend` | int64 | `1 if dayofweek >= 5` | Weekend response differences | Safe | Low |

### Growth Interactions (geospatial.py)

| Feature | Type | Computation | Rationale | Leakage | Priority |
|---------|------|-------------|-----------|---------|----------|
| `is_growing` | int64 | `1 if area_growth_abs > 0` | Binary flag: fire actively expanding | Safe | High |
| `growth_x_proximity` | float64 | `area_growth_rate / (dist_min + 1)` | Growing fire that is also close = dangerous | Safe | High |
| `speed_x_growth` | float64 | `closing_speed * area_growth_rate` | Fast-closing AND growing = highest threat | Safe | Medium |

## Redundant Features (dropped)

Per EDA correlation analysis (|r| > 0.95):

| Dropped | Kept | Reason |
|---------|------|--------|
| `relative_growth_0_5h` | `area_growth_rel_0_5h` | Identical (r ~ 1.0) |
| `closing_speed_abs_m_per_h` | `closing_speed_m_per_h` | Signed version more informative |
| `radial_growth_rate_m_per_h` | `radial_growth_m` | High correlation, keep the simpler one |

## Final Feature Count

- Base features (after dropping redundant): 31
- Engineered features: 16
- **Total: 47 features**

## Feature Selection Results

Populated after `notebooks/04_feature_analysis.ipynb` execution with mutual information scores and importance ranking.
