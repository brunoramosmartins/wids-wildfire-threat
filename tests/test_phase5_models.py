"""Phase 5: survival, boosting, and evaluation extras."""

import numpy as np
import pandas as pd

from src.features.selection import (
    compare_top_n_feature_sets,
    permutation_importance_ranking,
    recursive_feature_elimination_top,
)
from src.models.baselines import get_random_forest_baseline
from src.models.boosting import (
    get_catboost_model,
    get_lightgbm_model,
    get_xgboost_model,
)
from src.models.evaluate import compute_metrics
from src.models.survival import (
    get_cox_ph_model,
    get_gbs_model,
    get_rsf_model,
)


def _surv_data(n: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(99)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.uniform(-1, 2, n),
        }
    )
    y = pd.DataFrame(
        {
            "time_to_hit_hours": rng.uniform(0.5, 72, n),
            "event": rng.choice([0, 1], n, p=[0.45, 0.55]),
        }
    )
    return X, y


def test_cox_ph_horizon_predict() -> None:
    X, y = _surv_data()
    m = get_cox_ph_model(penalizer=0.5)
    m.fit(X, y)
    p = m.predict_proba_horizons(X)
    assert set(p.columns) == {"prob_12h", "prob_24h", "prob_48h", "prob_72h"}
    assert len(p) == len(X)
    assert (p["prob_24h"] >= p["prob_12h"] - 1e-8).all()


def test_cox_ph_predict_row_count_matches_large_n() -> None:
    """Regression: lifelines may return (n_times, n_samples); output must be n_samples rows."""
    X, y = _surv_data(200)
    m = get_cox_ph_model(penalizer=1.0)
    m.fit(X, y)
    p = m.predict_proba_horizons(X)
    assert len(p) == len(X)


def test_rsf_gbs_horizon_predict() -> None:
    X, y = _surv_data(100)
    for factory in (lambda: get_rsf_model(n_estimators=30), lambda: get_gbs_model(n_estimators=40)):
        m = factory()
        m.fit(X, y)
        p = m.predict_proba_horizons(X)
        assert p.shape[0] == len(X)
        assert (p["prob_72h"] >= p["prob_48h"] - 1e-8).all()


def test_boosting_horizon_fit() -> None:
    X, y = _surv_data(80)
    for factory in (
        lambda: get_xgboost_model(n_estimators=30, max_depth=3),
        lambda: get_lightgbm_model(n_estimators=40, max_depth=4),
        lambda: get_catboost_model(iterations=40, depth=4),
    ):
        m = factory()
        m.fit(X, y)
        p = m.predict_proba_horizons(X)
        metrics = compute_metrics(y, p)
        assert "brier_mean" in metrics


def test_selection_rfe_and_permutation() -> None:
    X, y = _surv_data(150)
    rank_df = permutation_importance_ranking(X, y, n_repeats=3)
    assert len(rank_df) == X.shape[1]
    rfe = recursive_feature_elimination_top(X, y, n_features_to_select=2, step=1)
    assert len(rfe) == 2


def test_compare_top_n_feature_sets() -> None:
    X, y = _surv_data(90)
    ranking = list(X.columns)
    out = compare_top_n_feature_sets(
        X,
        y,
        ranking,
        lambda: get_random_forest_baseline(),
        n_list=(2, 3),
        n_splits=3,
    )
    assert 2 in out and 3 in out
    assert "brier_mean_cv" in out[2]
