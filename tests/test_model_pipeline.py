"""Tests for model training and prediction pipeline."""

import numpy as np
import pandas as pd

from src.models.baselines import (
    get_kaplan_meier_baseline,
    get_logistic_baseline,
    get_random_forest_baseline,
)
from src.models.evaluate import compute_metrics


def _make_train_data(
    n: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic training data."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "dist_min": rng.uniform(100, 50000, n),
            "speed": rng.uniform(-100, 300, n),
            "alignment": rng.uniform(0, 1, n),
        }
    )
    y = pd.DataFrame(
        {
            "time_to_hit_hours": rng.uniform(0, 72, n),
            "event": rng.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )
    return X, y


# --- baselines ---


def test_kaplan_meier_baseline_fit_predict() -> None:
    X, y = _make_train_data()
    model = get_kaplan_meier_baseline()
    model.fit(X, y)
    preds = model.predict_proba_horizons(X)
    assert set(preds.columns) == {"prob_12h", "prob_24h", "prob_48h", "prob_72h"}
    assert len(preds) == len(X)
    # KM: all rows get the same value per horizon
    assert preds["prob_12h"].nunique() == 1


def test_kaplan_meier_monotonicity() -> None:
    X, y = _make_train_data()
    model = get_kaplan_meier_baseline()
    model.fit(X, y)
    preds = model.predict_proba_horizons(X)
    assert preds["prob_12h"].iloc[0] <= preds["prob_24h"].iloc[0]
    assert preds["prob_24h"].iloc[0] <= preds["prob_48h"].iloc[0]
    assert preds["prob_48h"].iloc[0] <= preds["prob_72h"].iloc[0]


def test_logistic_baseline_fit_predict() -> None:
    X, y = _make_train_data()
    model = get_logistic_baseline()
    model.fit(X, y)
    preds = model.predict_proba_horizons(X)
    assert set(preds.columns) == {"prob_12h", "prob_24h", "prob_48h", "prob_72h"}
    assert len(preds) == len(X)
    # Probabilities in [0, 1]
    for col in preds.columns:
        assert preds[col].min() >= 0
        assert preds[col].max() <= 1


def test_random_forest_baseline_fit_predict() -> None:
    X, y = _make_train_data()
    model = get_random_forest_baseline()
    model.fit(X, y)
    preds = model.predict_proba_horizons(X)
    assert set(preds.columns) == {"prob_12h", "prob_24h", "prob_48h", "prob_72h"}
    assert len(preds) == len(X)


def test_multi_horizon_monotonicity() -> None:
    """Predictions must be monotonically non-decreasing across horizons."""
    X, y = _make_train_data(100)
    model = get_logistic_baseline()
    model.fit(X, y)
    preds = model.predict_proba_horizons(X)
    assert (preds["prob_24h"] >= preds["prob_12h"] - 1e-9).all()
    assert (preds["prob_48h"] >= preds["prob_24h"] - 1e-9).all()
    assert (preds["prob_72h"] >= preds["prob_48h"] - 1e-9).all()


# --- evaluate ---


def test_compute_metrics_returns_brier() -> None:
    X, y = _make_train_data()
    model = get_logistic_baseline()
    model.fit(X, y)
    preds = model.predict_proba_horizons(X)
    metrics = compute_metrics(y, preds)
    assert "brier_mean" in metrics
    assert 0 <= metrics["brier_mean"] <= 1
    for h in [12, 24, 48, 72]:
        assert f"brier_{h}h" in metrics
