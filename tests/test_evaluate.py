"""Tests for the WiDS 2026 Hybrid Score and its components."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.evaluate import (
    BRIER_WEIGHTS,
    SCORED_HORIZONS,
    censor_aware_brier_at_horizon,
    compute_metrics,
    harrell_c_index,
    hybrid_score,
    weighted_brier_score,
)

PROB_COLS = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]


def _mk_y(events: list[int], times: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"event": events, "time_to_hit_hours": times})


def _mk_preds(rows: list[list[float]]) -> pd.DataFrame:
    arr = np.asarray(rows, dtype=float)
    return pd.DataFrame(arr, columns=PROB_COLS)


# -----------------------------------------------------------------------------
# censor_aware_brier_at_horizon
# -----------------------------------------------------------------------------


def test_censor_aware_brier_excludes_censored_before_horizon() -> None:
    """A sample censored at T<H must be dropped from the Brier at H."""
    # Sample 0: censored at T=10 -> T<24 -> EXCLUDED
    # Sample 1: hit at T=5 -> label=1
    # Sample 2: censored at T=50 -> T>24 -> label=0
    # Sample 3: hit at T=30 -> T>24 -> label=0
    y = _mk_y(events=[0, 1, 0, 1], times=[10.0, 5.0, 50.0, 30.0])
    preds = np.array([0.5, 1.0, 0.0, 0.0])
    b = censor_aware_brier_at_horizon(y, preds, 24.0)
    assert b == 0.0


def test_censor_aware_brier_counts_hit_within_horizon_as_one() -> None:
    # Sample 0: hit@10, H=24 -> label=1, pred=1.0 -> Brier=0
    # Sample 1: hit@40>24 -> label=0, pred=0.0 -> Brier=0
    y = _mk_y(events=[1, 1], times=[10.0, 40.0])
    preds = np.array([1.0, 0.0])
    b = censor_aware_brier_at_horizon(y, preds, 24.0)
    assert b == 0.0


def test_censor_aware_brier_counts_censored_after_horizon_as_zero() -> None:
    y = _mk_y(events=[0], times=[50.0])  # censored at T=50, H=24 -> label=0, included
    preds = np.array([0.3])
    b = censor_aware_brier_at_horizon(y, preds, 24.0)
    assert abs(b - 0.09) < 1e-9


def test_censor_aware_brier_all_excluded_returns_nan() -> None:
    y = _mk_y(events=[0, 0], times=[5.0, 10.0])  # both censored before H=24
    preds = np.array([0.5, 0.5])
    b = censor_aware_brier_at_horizon(y, preds, 24.0)
    assert np.isnan(b)


# -----------------------------------------------------------------------------
# weighted_brier_score
# -----------------------------------------------------------------------------


def test_weighted_brier_weights_sum_to_one() -> None:
    assert abs(sum(BRIER_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(BRIER_WEIGHTS) == set(SCORED_HORIZONS)


def test_weighted_brier_perfect_predictions() -> None:
    # 3 fires: hit @ 18h, hit @ 40h, censored @ 72h (survived)
    y = _mk_y(events=[1, 1, 0], times=[18.0, 40.0, 72.0])
    # Perfect cumulative P(hit by T): monotone and correct
    preds = _mk_preds(
        [
            [0.0, 1.0, 1.0, 1.0],  # hit@18 -> P(hit by 24)=1
            [0.0, 0.0, 1.0, 1.0],  # hit@40 -> P(hit by 48)=1
            [0.0, 0.0, 0.0, 0.0],  # censored @72 -> never predicted
        ]
    )
    wb, per_h = weighted_brier_score(y, preds)
    assert per_h[24] == 0.0
    assert per_h[48] == 0.0
    assert per_h[72] == 0.0
    assert wb == 0.0


def test_weighted_brier_weights_applied() -> None:
    # Construct predictions so Brier@24=0.0, Brier@48=0.1, Brier@72=0.0
    # Expected weighted = 0.3*0 + 0.4*0.1 + 0.3*0 = 0.04
    y = _mk_y(events=[1], times=[50.0])  # hit@50: label_24=0, label_48=0, label_72=1
    preds = _mk_preds([[0.0, 0.0, 0.316227766, 1.0]])  # pred_48 = sqrt(0.1) → sq-err = 0.1
    wb, per_h = weighted_brier_score(y, preds)
    assert abs(per_h[24] - 0.0) < 1e-9
    assert abs(per_h[48] - 0.1) < 1e-6
    assert abs(per_h[72] - 0.0) < 1e-9
    assert abs(wb - 0.04) < 1e-6


# -----------------------------------------------------------------------------
# harrell_c_index
# -----------------------------------------------------------------------------


def test_c_index_perfect_ranking() -> None:
    y = _mk_y(events=[1, 1, 1], times=[10.0, 20.0, 30.0])
    # Higher prob_72h = sooner event. Perfect ranking = prob_72h sorted desc by event time
    preds = _mk_preds(
        [
            [0.1, 0.3, 0.6, 0.9],  # earliest hit -> highest risk
            [0.05, 0.2, 0.4, 0.6],
            [0.02, 0.1, 0.2, 0.3],
        ]
    )
    ci = harrell_c_index(y, preds)
    assert abs(ci - 1.0) < 1e-9


def test_c_index_inverted_ranking() -> None:
    y = _mk_y(events=[1, 1, 1], times=[10.0, 20.0, 30.0])
    # Inverted: latest hit gets highest risk -> C-index = 0
    preds = _mk_preds(
        [
            [0.02, 0.1, 0.2, 0.3],
            [0.05, 0.2, 0.4, 0.6],
            [0.1, 0.3, 0.6, 0.9],
        ]
    )
    ci = harrell_c_index(y, preds)
    assert abs(ci - 0.0) < 1e-9


# -----------------------------------------------------------------------------
# hybrid_score
# -----------------------------------------------------------------------------


def test_hybrid_score_formula() -> None:
    """Hybrid = 0.3 * C + 0.7 * (1 - WB).

    Perfect predictions + strictly ordered risks -> Hybrid = 1.0.
    """
    y = _mk_y(events=[1, 1, 0], times=[18.0, 40.0, 72.0])
    # Brier-perfect AT scored horizons (24/48/72) AND strictly ordered risk via prob_72h
    preds = _mk_preds(
        [
            [0.0, 1.0, 1.0, 0.99],  # earliest hit, risk=0.99
            [0.0, 0.0, 1.0, 0.9],  # mid hit, risk=0.9
            [0.0, 0.0, 0.0, 0.0],  # censored, risk=0.0
        ]
    )
    hybrid, comps = hybrid_score(y, preds)
    assert abs(comps["c_index"] - 1.0) < 1e-9
    # Brier not exactly zero at 72h because we nudged prob_72h from 1.0
    # to 0.99/0.9 to create strict ranking. Small penalty expected.
    assert comps["weighted_brier"] > 0.0
    assert abs(comps["hybrid_score"] - hybrid) < 1e-12
    # Lower bound sanity: hybrid must be well above 0.5
    assert hybrid > 0.7


def test_hybrid_score_maximum_is_one() -> None:
    """Hybrid in [0, 1] for valid inputs."""
    y = _mk_y(events=[1, 0, 1, 0], times=[12.0, 72.0, 48.0, 72.0])
    preds = _mk_preds(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    hybrid, _ = hybrid_score(y, preds)
    assert 0.0 <= hybrid <= 1.0


# -----------------------------------------------------------------------------
# compute_metrics
# -----------------------------------------------------------------------------


def test_compute_metrics_exposes_hybrid() -> None:
    y = _mk_y(events=[1, 0, 1, 1, 0], times=[18.0, 72.0, 40.0, 60.0, 30.0])
    preds = _mk_preds(
        [
            [0.1, 0.6, 0.9, 0.95],
            [0.05, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.7, 0.85],
            [0.2, 0.4, 0.6, 0.8],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    metrics = compute_metrics(y, preds)
    assert "hybrid_score" in metrics
    assert "c_index" in metrics
    assert "weighted_brier" in metrics
    assert "brier_24h_ca" in metrics
    assert "brier_48h_ca" in metrics
    assert "brier_72h_ca" in metrics
    # Backward-compat diagnostics
    assert "brier_mean" in metrics
    assert "brier_12h_naive" in metrics
    assert 0.0 <= metrics["hybrid_score"] <= 1.0
