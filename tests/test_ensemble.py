"""Tests for Phase 6 ensemble methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import (
    BlendingEnsemble,
    StackingEnsemble,
    blending_ensemble,
    enforce_monotonicity,
    optimize_weights,
    stacking_ensemble,
    weighted_average,
)

PROB_COLS = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]


def _sample_preds(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p12 = rng.uniform(0.0, 0.3, n)
    p24 = p12 + rng.uniform(0.0, 0.15, n)
    p48 = p24 + rng.uniform(0.0, 0.15, n)
    p72 = p48 + rng.uniform(0.0, 0.15, n)
    return pd.DataFrame(
        np.clip(np.stack([p12, p24, p48, p72], axis=1), 0.0, 1.0),
        columns=PROB_COLS,
    )


def _sample_y(n: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time_to_hit_hours": rng.uniform(0.0, 72.0, n),
            "event": rng.integers(0, 2, n),
        }
    )


# --- enforce_monotonicity ---


def test_enforce_monotonicity_fixes_violation() -> None:
    df = pd.DataFrame(
        {
            "prob_12h": [0.9, 0.1],
            "prob_24h": [0.3, 0.2],  # violates at row 0
            "prob_48h": [0.5, 0.3],
            "prob_72h": [0.7, 0.4],
        }
    )
    out = enforce_monotonicity(df)
    assert (out["prob_24h"] >= out["prob_12h"] - 1e-9).all()
    assert (out["prob_48h"] >= out["prob_24h"] - 1e-9).all()
    assert (out["prob_72h"] >= out["prob_48h"] - 1e-9).all()
    # Row 0 had prob_24h=0.3 < prob_12h=0.9 → should lift to 0.9
    assert out.loc[0, "prob_24h"] == 0.9


def test_enforce_monotonicity_clips_range() -> None:
    df = pd.DataFrame(
        {
            "prob_12h": [-0.1, 1.5],
            "prob_24h": [0.2, 1.1],
            "prob_48h": [0.3, 1.2],
            "prob_72h": [0.4, 1.3],
        }
    )
    out = enforce_monotonicity(df)
    assert out.min().min() >= 0.0
    assert out.max().max() <= 1.0


# --- weighted_average ---


def test_weighted_average_single_model_equals_input() -> None:
    preds = _sample_preds(10, seed=0)
    out = weighted_average([preds], [1.0])
    np.testing.assert_allclose(out[PROB_COLS].values, preds[PROB_COLS].values, atol=1e-9)


def test_weighted_average_three_models_same_input() -> None:
    preds = _sample_preds(10, seed=0)
    out = weighted_average([preds, preds, preds], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out[PROB_COLS].values, preds[PROB_COLS].values, atol=1e-9)


def test_weighted_average_normalizes_weights() -> None:
    p1 = _sample_preds(10, seed=0)
    p2 = _sample_preds(10, seed=1)
    out1 = weighted_average([p1, p2], [1.0, 1.0])
    out2 = weighted_average([p1, p2], [0.5, 0.5])  # identical after normalization
    np.testing.assert_allclose(out1[PROB_COLS].values, out2[PROB_COLS].values, atol=1e-9)


def test_weighted_average_rejects_negative_weights() -> None:
    p = _sample_preds(5)
    with pytest.raises(ValueError, match="non-negative"):
        weighted_average([p, p], [-0.5, 0.5])


def test_weighted_average_rejects_zero_sum() -> None:
    p = _sample_preds(5)
    with pytest.raises(ValueError, match="positive value"):
        weighted_average([p, p], [0.0, 0.0])


def test_weighted_average_enforces_monotonicity() -> None:
    # Craft a non-monotone prediction
    p_bad = pd.DataFrame(
        {"prob_12h": [0.9], "prob_24h": [0.1], "prob_48h": [0.2], "prob_72h": [0.3]}
    )
    out = weighted_average([p_bad], [1.0])
    assert out.loc[0, "prob_24h"] >= out.loc[0, "prob_12h"]


# --- optimize_weights ---


def test_optimize_weights_returns_simplex() -> None:
    p1 = _sample_preds(50, seed=0)
    p2 = _sample_preds(50, seed=1)
    y = _sample_y(50)
    w = optimize_weights([p1, p2], y)
    assert w.shape == (2,)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-9).all()


def test_optimize_weights_prefers_better_model() -> None:
    # Build "good" preds that match labels at 72h and "bad" random preds
    rng = np.random.default_rng(42)
    n = 60
    y = pd.DataFrame(
        {
            "time_to_hit_hours": rng.uniform(0.0, 72.0, n),
            "event": rng.integers(0, 2, n),
        }
    )
    labels72 = ((y["event"] == 1) & (y["time_to_hit_hours"] <= 72)).astype(float).values
    good = pd.DataFrame(
        {
            "prob_12h": labels72 * 0.3,
            "prob_24h": labels72 * 0.5,
            "prob_48h": labels72 * 0.8,
            "prob_72h": labels72,
        }
    )
    bad = pd.DataFrame({c: rng.uniform(0.0, 1.0, n) for c in PROB_COLS})
    w = optimize_weights([good, bad], y)
    assert w[0] > w[1], f"expected good > bad; got w={w}"


# --- stacking ---


def test_stacking_ensemble_fit_predict_shape() -> None:
    p1 = _sample_preds(30, seed=0)
    p2 = _sample_preds(30, seed=1)
    y = _sample_y(30)
    model = StackingEnsemble(random_state=0).fit([p1, p2], y)
    test_p1 = _sample_preds(10, seed=2)
    test_p2 = _sample_preds(10, seed=3)
    out = model.predict_proba_horizons([test_p1, test_p2])
    assert list(out.columns) == PROB_COLS
    assert len(out) == 10
    assert out.min().min() >= 0.0
    assert out.max().max() <= 1.0


def test_stacking_enforces_monotonicity() -> None:
    p1 = _sample_preds(30, seed=0)
    y = _sample_y(30)
    model = StackingEnsemble(random_state=0).fit([p1], y)
    out = model.predict_proba_horizons([p1])
    assert (out["prob_24h"] >= out["prob_12h"] - 1e-9).all()
    assert (out["prob_72h"] >= out["prob_48h"] - 1e-9).all()


def test_stacking_rejects_wrong_n_models() -> None:
    p1 = _sample_preds(20, seed=0)
    p2 = _sample_preds(20, seed=1)
    y = _sample_y(20)
    model = StackingEnsemble(random_state=0).fit([p1, p2], y)
    with pytest.raises(ValueError, match="expected 2 models"):
        model.predict_proba_horizons([p1])


def test_stacking_functional_alias() -> None:
    p1 = _sample_preds(30, seed=0)
    p2 = _sample_preds(30, seed=1)
    y = _sample_y(30)
    out = stacking_ensemble([p1, p2], y)
    assert list(out.columns) == PROB_COLS
    assert len(out) == 30


# --- blending ---


def test_blending_ensemble_fit_predict_shape() -> None:
    p1 = _sample_preds(30, seed=0)
    p2 = _sample_preds(30, seed=1)
    y = _sample_y(30)
    model = BlendingEnsemble(random_state=0).fit([p1, p2], y)
    test_p1 = _sample_preds(8, seed=5)
    test_p2 = _sample_preds(8, seed=6)
    out = model.predict_proba_horizons([test_p1, test_p2])
    assert list(out.columns) == PROB_COLS
    assert len(out) == 8


def test_blending_functional_alias() -> None:
    p1 = _sample_preds(30, seed=0)
    p2 = _sample_preds(30, seed=1)
    y = _sample_y(30)
    test_p1 = _sample_preds(10, seed=2)
    test_p2 = _sample_preds(10, seed=3)
    out = blending_ensemble([p1, p2], y, [test_p1, test_p2])
    assert list(out.columns) == PROB_COLS
    assert len(out) == 10
