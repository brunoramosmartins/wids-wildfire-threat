"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd

from src.features.geospatial import compute_geospatial_features
from src.features.selection import find_correlated_pairs, rank_mutual_information
from src.features.temporal import compute_temporal_features


def _make_sample_df() -> pd.DataFrame:
    """Create a minimal DataFrame with all required columns."""
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "dist_min_ci_0_5h": [1000.0, 5000.0, 20000.0, 500.0, 50000.0],
            "closing_speed_m_per_h": [100.0, -50.0, 200.0, 300.0, 0.0],
            "projected_advance_m": [500.0, -250.0, 1000.0, 1500.0, 0.0],
            "alignment_abs": [0.9, 0.2, 0.7, 0.95, 0.1],
            "area_growth_abs_0_5h": [10.0, 0.0, 5.0, 20.0, 0.0],
            "area_growth_rate_ha_per_h": [2.0, 0.0, 1.0, 4.0, 0.0],
            "event_start_hour": [14, 3, 10, 22, 7],
            "event_start_month": [7, 1, 8, 6, 12],
            "event_start_dayofweek": [0, 5, 2, 6, 4],
        }
    )


# --- geospatial ---


def test_geospatial_output_columns() -> None:
    df = _make_sample_df()
    result = compute_geospatial_features(df)
    expected = {
        "threat_score",
        "projected_arrival_h",
        "is_closing",
        "dist_min_log",
        "dist_bin",
        "advance_ratio",
        "alignment_x_speed",
        "is_growing",
        "growth_x_proximity",
        "speed_x_growth",
    }
    assert set(result.columns) == expected


def test_is_closing_binary() -> None:
    df = _make_sample_df()
    result = compute_geospatial_features(df)
    assert set(result["is_closing"].unique()).issubset({0, 1})


def test_is_growing_binary() -> None:
    df = _make_sample_df()
    result = compute_geospatial_features(df)
    assert set(result["is_growing"].unique()).issubset({0, 1})


def test_projected_arrival_clipped() -> None:
    df = _make_sample_df()
    result = compute_geospatial_features(df)
    assert result["projected_arrival_h"].max() <= 200
    assert result["projected_arrival_h"].min() >= 0


def test_threat_score_positive_when_closing() -> None:
    df = _make_sample_df()
    result = compute_geospatial_features(df)
    closing_mask = df["closing_speed_m_per_h"] > 0
    assert (result.loc[closing_mask, "threat_score"] > 0).all()


# --- temporal ---


def test_temporal_output_columns() -> None:
    df = _make_sample_df()
    result = compute_temporal_features(df)
    expected = {
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "is_daytime",
        "is_weekend",
    }
    assert set(result.columns) == expected


def test_cyclical_range() -> None:
    df = _make_sample_df()
    result = compute_temporal_features(df)
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert result[col].min() >= -1.0
        assert result[col].max() <= 1.0


def test_is_daytime_correct() -> None:
    df = _make_sample_df()
    result = compute_temporal_features(df)
    # hour=14 -> daytime, hour=3 -> night, hour=10 -> day, hour=22 -> night, hour=7 -> day
    expected = [1, 0, 1, 0, 1]
    assert list(result["is_daytime"]) == expected


def test_is_weekend_correct() -> None:
    df = _make_sample_df()
    result = compute_temporal_features(df)
    # dow: 0=Mon, 5=Sat, 2=Wed, 6=Sun, 4=Fri
    expected = [0, 1, 0, 1, 0]
    assert list(result["is_weekend"]) == expected


# --- selection ---


def test_rank_mutual_information_returns_sorted() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    target = pd.Series(df["a"] * 2 + rng.normal(0, 0.1, size=50))
    scores = rank_mutual_information(df, target)
    assert scores.index[0] == "a"
    assert len(scores) == 2


def test_find_correlated_pairs() -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.1, 2.1, 3.1, 4.1, 5.1],  # near-perfect correlation with x
            "z": [5.0, 4.0, 3.0, 2.0, 1.0],  # negative correlation
        }
    )
    pairs = find_correlated_pairs(df, threshold=0.95)
    pair_cols = [(p[0], p[1]) for p in pairs]
    assert ("x", "y") in pair_cols or ("y", "x") in pair_cols
