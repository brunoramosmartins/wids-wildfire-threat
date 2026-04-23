"""Tests for Phase 6.5 additions.

Covers:
- Weibull & LogNormal AFT horizon models
- Isotonic horizon calibrator
- Seed ensembling wrapper
- Monotone-constraint vector builder
- Repeated stratified CV runner
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.aft import get_lognormal_aft_model, get_weibull_aft_model
from src.models.baselines import get_logistic_baseline
from src.models.boosting import (
    MONOTONE_CONSTRAINTS_DEFAULT,
    get_xgboost_model,
    monotone_vector,
)
from src.models.calibration import IsotonicHorizonCalibrator
from src.models.seed_ensemble import SeedEnsembleWrapper
from src.validation.repeated_cv import repeated_stratified_cv

PROB_COLS = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]


def _sample_train(n: int = 80, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "dist_min_ci_0_5h": rng.uniform(500, 40000, n),
            "closing_speed_m_per_h": rng.uniform(-200, 400, n),
            "alignment_abs": rng.uniform(0, 1, n),
            "area_first_ha": rng.uniform(0.1, 500, n),
        }
    )
    # Generate event/time with a mild relationship to features
    score = -X["dist_min_ci_0_5h"] / 5000 + X["closing_speed_m_per_h"] / 200
    y_event = rng.uniform(size=n) < (0.2 + 0.3 * (score > score.median()))
    y_event = y_event.astype(int)
    y_time = rng.uniform(0.5, 72.0, n)
    return X, pd.DataFrame({"time_to_hit_hours": y_time, "event": y_event})


# -----------------------------------------------------------------------------
# AFT
# -----------------------------------------------------------------------------


def test_weibull_aft_fit_predict() -> None:
    X, y = _sample_train()
    m = get_weibull_aft_model(penalizer=0.05)
    m.fit(X, y)
    preds = m.predict_proba_horizons(X.iloc[:10])
    assert list(preds.columns) == PROB_COLS
    assert len(preds) == 10
    assert preds.min().min() >= 0
    assert preds.max().max() <= 1


def test_weibull_aft_monotone_across_horizons() -> None:
    X, y = _sample_train()
    m = get_weibull_aft_model()
    m.fit(X, y)
    preds = m.predict_proba_horizons(X.iloc[:20])
    assert (preds["prob_24h"] >= preds["prob_12h"] - 1e-9).all()
    assert (preds["prob_72h"] >= preds["prob_48h"] - 1e-9).all()


def test_lognormal_aft_fit_predict() -> None:
    X, y = _sample_train()
    m = get_lognormal_aft_model(penalizer=0.05)
    m.fit(X, y)
    preds = m.predict_proba_horizons(X.iloc[:10])
    assert list(preds.columns) == PROB_COLS
    assert len(preds) == 10


# -----------------------------------------------------------------------------
# Isotonic calibration
# -----------------------------------------------------------------------------


def test_isotonic_calibrator_fit_transform_preserves_shape() -> None:
    _, y = _sample_train()
    n = len(y)
    rng = np.random.default_rng(1)
    # Make predictions that are decent but a bit off
    base = rng.uniform(0, 1, n)
    oof = pd.DataFrame(
        {
            "prob_12h": base * 0.2,
            "prob_24h": base * 0.4,
            "prob_48h": base * 0.7,
            "prob_72h": base,
        }
    )
    calib = IsotonicHorizonCalibrator(min_events_per_horizon=3).fit(oof, y)
    out = calib.transform(oof)
    assert list(out.columns) == PROB_COLS
    assert len(out) == n
    assert out.min().min() >= 0.0
    assert out.max().max() <= 1.0


def test_isotonic_calibrator_enforces_monotonicity() -> None:
    _, y = _sample_train()
    n = len(y)
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, n)
    oof = pd.DataFrame(
        {
            "prob_12h": base * 0.3,
            "prob_24h": base * 0.5,
            "prob_48h": base * 0.7,
            "prob_72h": base,
        }
    )
    calib = IsotonicHorizonCalibrator().fit(oof, y)
    out = calib.transform(oof)
    assert (out["prob_24h"] >= out["prob_12h"] - 1e-9).all()
    assert (out["prob_72h"] >= out["prob_48h"] - 1e-9).all()


def test_isotonic_calibrator_summary_reports_status() -> None:
    _, y = _sample_train()
    n = len(y)
    oof = pd.DataFrame({c: np.full(n, 0.3) for c in PROB_COLS})
    calib = IsotonicHorizonCalibrator().fit(oof, y)
    s = calib.summary()
    for h in [12, 24, 48, 72]:
        assert h in s


# -----------------------------------------------------------------------------
# SeedEnsembleWrapper
# -----------------------------------------------------------------------------


def test_seed_ensemble_averages_predictions() -> None:
    X, y = _sample_train(n=60)

    def factory():  # noqa: D401
        return get_logistic_baseline()

    model = SeedEnsembleWrapper(factory, seeds=[1, 2, 3]).fit(X, y)
    preds = model.predict_proba_horizons(X)
    assert list(preds.columns) == PROB_COLS
    assert len(preds) == len(X)


def test_seed_ensemble_monotone() -> None:
    X, y = _sample_train(n=60)

    def factory():  # noqa: D401
        return get_logistic_baseline()

    model = SeedEnsembleWrapper(factory, seeds=[1, 2]).fit(X, y)
    preds = model.predict_proba_horizons(X)
    assert (preds["prob_24h"] >= preds["prob_12h"] - 1e-9).all()
    assert (preds["prob_72h"] >= preds["prob_48h"] - 1e-9).all()


# -----------------------------------------------------------------------------
# Monotone constraints
# -----------------------------------------------------------------------------


def test_monotone_defaults_known_signs() -> None:
    """Key features must have the expected monotone sign."""
    assert MONOTONE_CONSTRAINTS_DEFAULT["dist_min_ci_0_5h"] == -1
    assert MONOTONE_CONSTRAINTS_DEFAULT["closing_speed_m_per_h"] == +1
    assert MONOTONE_CONSTRAINTS_DEFAULT["alignment_abs"] == +1
    assert MONOTONE_CONSTRAINTS_DEFAULT["is_closing"] == +1


def test_monotone_vector_maps_unknown_to_zero() -> None:
    feats = ["dist_min_ci_0_5h", "unknown_feature", "closing_speed_m_per_h"]
    v = monotone_vector(feats)
    assert v == [-1, 0, +1]


def test_monotone_vector_overrides_win() -> None:
    feats = ["dist_min_ci_0_5h"]
    v = monotone_vector(feats, overrides={"dist_min_ci_0_5h": 0})
    assert v == [0]


def test_xgboost_accepts_monotone_constraints() -> None:
    X, y = _sample_train(n=50)
    model = get_xgboost_model(
        n_estimators=30,
        max_depth=3,
        feature_names=X.columns.tolist(),
    )
    model.fit(X, y)
    preds = model.predict_proba_horizons(X)
    assert list(preds.columns) == PROB_COLS


# -----------------------------------------------------------------------------
# Repeated CV
# -----------------------------------------------------------------------------


def test_repeated_cv_runs_and_aggregates() -> None:
    X, y = _sample_train(n=80)

    def factory():  # noqa: D401
        return get_logistic_baseline()

    result = repeated_stratified_cv(factory, X, y, n_splits=3, n_repeats=2, base_seed=42)
    assert result["n_evaluations"] == 6  # 3 folds × 2 repeats
    assert 0.0 <= result["hybrid_score_mean"] <= 1.0
    assert result["hybrid_score_std"] >= 0.0
    assert 0.0 <= result["weighted_brier_mean"] <= 1.0
