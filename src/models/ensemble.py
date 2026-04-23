"""Ensemble model implementations (Phase 6).

Combines top Phase 5/6 models via:
- ``weighted_average``: convex combination with weights summing to 1.
- ``optimize_weights``: scipy SLSQP minimizes mean CV Brier on OOF preds.
- ``StackingEnsemble``: per-horizon meta-learner trained on OOF predictions.
- ``BlendingEnsemble``: single holdout split; meta-learner on holdout preds.

All ensembles operate on per-horizon probability DataFrames with columns
``prob_12h, prob_24h, prob_48h, prob_72h`` and enforce monotonicity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from src.models.baselines import HORIZONS

__all__ = [
    "BlendingEnsemble",
    "StackingEnsemble",
    "enforce_monotonicity",
    "optimize_weights",
    "weighted_average",
]

PROB_COLS = [f"prob_{h}h" for h in HORIZONS]


def enforce_monotonicity(preds: pd.DataFrame) -> pd.DataFrame:
    """Ensure P(12h) <= P(24h) <= P(48h) <= P(72h) row-wise."""
    out = preds.copy()
    for i in range(1, len(HORIZONS)):
        prev_c = f"prob_{HORIZONS[i - 1]}h"
        curr_c = f"prob_{HORIZONS[i]}h"
        out[curr_c] = np.maximum(out[curr_c].values, out[prev_c].values)
    return out.clip(0.0, 1.0)


def weighted_average(
    predictions: list[pd.DataFrame],
    weights: list[float],
) -> pd.DataFrame:
    """Combine per-horizon probability DataFrames by convex weighted average.

    Weights are normalized to sum to 1. Monotonicity enforced on output.
    """
    if not predictions:
        raise ValueError("predictions list is empty")
    if len(predictions) != len(weights):
        raise ValueError(f"got {len(predictions)} predictions but {len(weights)} weights")
    w = np.asarray(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    w = w / total

    # Check all have same shape
    n_rows = len(predictions[0])
    for i, p in enumerate(predictions[1:], start=1):
        if len(p) != n_rows:
            raise ValueError(f"prediction {i} has {len(p)} rows, expected {n_rows}")

    agg = np.zeros((n_rows, len(PROB_COLS)), dtype=float)
    for pred, wi in zip(predictions, w, strict=True):
        agg += wi * pred[PROB_COLS].values

    result = pd.DataFrame(agg, columns=PROB_COLS, index=predictions[0].index)
    return enforce_monotonicity(result)


def _labels_from_y(y: pd.DataFrame) -> np.ndarray:
    """Binary labels at each horizon (hit within H). Shape (n, 4)."""
    T = y["time_to_hit_hours"].values
    E = y["event"].values
    return np.column_stack([((E == 1) & (T <= h)).astype(np.int32) for h in HORIZONS])


def _brier_of_weighted(
    weights: np.ndarray,
    preds_array: np.ndarray,  # shape (n_models, n_samples, n_horizons)
    labels: np.ndarray,  # shape (n_samples, n_horizons), binary
) -> float:
    """Mean naive-Brier across horizons (legacy; used when objective='brier')."""
    w = weights / weights.sum()
    combined = np.einsum("i,ijk->jk", w, preds_array)
    combined = np.maximum.accumulate(combined, axis=1).clip(0.0, 1.0)
    brier_per_h = np.mean((combined - labels) ** 2, axis=0)
    return float(np.mean(brier_per_h))


def _neg_hybrid_of_weighted(
    weights: np.ndarray,
    preds_array: np.ndarray,  # (n_models, n_samples, n_horizons)
    y: pd.DataFrame,
) -> float:
    """Negative Hybrid Score of a convex combination of model preds.

    We negate so SLSQP (which minimizes) maximizes Hybrid.
    Uses the official Hybrid Score from src.models.evaluate.
    """
    # Avoid circular import at module load
    from src.models.evaluate import hybrid_score as _hybrid

    w = weights / weights.sum()
    combined = np.einsum("i,ijk->jk", w, preds_array)
    combined = np.maximum.accumulate(combined, axis=1).clip(0.0, 1.0)
    pred_df = pd.DataFrame(combined, columns=PROB_COLS, index=y.index)
    h, _ = _hybrid(y, pred_df)
    return -float(h)


def optimize_weights(
    oof_predictions: list[pd.DataFrame],
    y: pd.DataFrame,
    seed: int = 42,
    objective: str = "hybrid",
) -> np.ndarray:
    """Find convex weights optimizing the chosen objective on OOF predictions.

    - ``objective='hybrid'``: MAXIMIZE Hybrid Score (official WiDS 2026 metric).
    - ``objective='brier'``: MINIMIZE mean naive-Brier (legacy behavior).

    Uses SLSQP with sum-to-one constraint and non-negativity bounds.
    Returns normalized weights (sum = 1).
    """
    n_models = len(oof_predictions)
    if n_models == 0:
        raise ValueError("no OOF predictions provided")
    if n_models == 1:
        return np.array([1.0])

    preds_array = np.stack([p[PROB_COLS].values for p in oof_predictions], axis=0)

    x0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    _ = seed  # symmetry only — SLSQP deterministic given x0

    if objective == "hybrid":
        fun = lambda w: _neg_hybrid_of_weighted(w, preds_array, y)  # noqa: E731
    elif objective == "brier":
        labels = _labels_from_y(y)
        fun = lambda w: _brier_of_weighted(w, preds_array, labels)  # noqa: E731
    else:
        raise ValueError(f"Unknown objective: {objective}")

    result = minimize(
        fun=fun,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    weights = np.clip(result.x, 0.0, 1.0)
    if weights.sum() <= 0:
        return x0
    return weights / weights.sum()


class StackingEnsemble:
    """Per-horizon logistic-regression meta-learner on OOF predictions.

    Fit on stacked OOF model probabilities + binary horizon labels.
    At predict time, consumes a list of horizon-probability DataFrames
    (same model order as training) and outputs final probabilities.
    """

    def __init__(self, random_state: int = 42, C: float = 1.0) -> None:
        self.random_state = random_state
        self.C = C
        self.meta_: dict[int, LogisticRegression] = {}
        self.n_models_: int = 0

    def fit(
        self,
        oof_predictions: list[pd.DataFrame],
        y: pd.DataFrame,
    ) -> StackingEnsemble:
        self.n_models_ = len(oof_predictions)
        labels = _labels_from_y(y)
        for i, h in enumerate(HORIZONS):
            col = f"prob_{h}h"
            # Stack this horizon across models → shape (n_samples, n_models)
            stacked = np.column_stack([p[col].values for p in oof_predictions])
            lr = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
            )
            lr.fit(stacked, labels[:, i])
            self.meta_[h] = lr
        return self

    def predict_proba_horizons(
        self,
        test_predictions: list[pd.DataFrame],
    ) -> pd.DataFrame:
        if len(test_predictions) != self.n_models_:
            raise ValueError(f"expected {self.n_models_} models, got {len(test_predictions)}")
        n_rows = len(test_predictions[0])
        out = pd.DataFrame(
            index=test_predictions[0].index,
            columns=PROB_COLS,
            dtype=float,
        )
        for h in HORIZONS:
            col = f"prob_{h}h"
            stacked = np.column_stack([p[col].values for p in test_predictions])
            out[col] = self.meta_[h].predict_proba(stacked)[:, 1]
        # Guard: fill any NaN (shouldn't happen) and enforce monotonicity
        out = out.fillna(0.0).astype(float)
        assert len(out) == n_rows
        return enforce_monotonicity(out)


class BlendingEnsemble:
    """Holdout-based blending.

    Fits base models on train split, meta-learner on holdout predictions.
    Lighter than stacking — single split rather than K-fold OOF.
    Use when training is expensive or OOF loop is impractical.
    """

    def __init__(self, random_state: int = 42, C: float = 1.0) -> None:
        self.random_state = random_state
        self.C = C
        self.meta_: dict[int, LogisticRegression] = {}
        self.n_models_: int = 0

    def fit(
        self,
        holdout_predictions: list[pd.DataFrame],
        holdout_y: pd.DataFrame,
    ) -> BlendingEnsemble:
        """Fit meta-learners on a single holdout slice's base predictions."""
        self.n_models_ = len(holdout_predictions)
        labels = _labels_from_y(holdout_y)
        for i, h in enumerate(HORIZONS):
            col = f"prob_{h}h"
            stacked = np.column_stack([p[col].values for p in holdout_predictions])
            lr = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
            )
            lr.fit(stacked, labels[:, i])
            self.meta_[h] = lr
        return self

    def predict_proba_horizons(
        self,
        test_predictions: list[pd.DataFrame],
    ) -> pd.DataFrame:
        if len(test_predictions) != self.n_models_:
            raise ValueError(f"expected {self.n_models_} models, got {len(test_predictions)}")
        out = pd.DataFrame(
            index=test_predictions[0].index,
            columns=PROB_COLS,
            dtype=float,
        )
        for h in HORIZONS:
            col = f"prob_{h}h"
            stacked = np.column_stack([p[col].values for p in test_predictions])
            out[col] = self.meta_[h].predict_proba(stacked)[:, 1]
        return enforce_monotonicity(out.fillna(0.0).astype(float))


# Back-compat alias functions (kept for the original roadmap API shape)


def stacking_ensemble(
    base_predictions: list[pd.DataFrame],
    target: pd.DataFrame,
    test_predictions: list[pd.DataFrame] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Functional wrapper: fit stacking meta-learner, return test predictions.

    If ``test_predictions`` is None, returns in-sample OOF preds reconstructed
    from base predictions (i.e. predictions on the same data used for stacking).
    """
    stack = StackingEnsemble(**kwargs).fit(base_predictions, target)
    if test_predictions is None:
        test_predictions = base_predictions
    return stack.predict_proba_horizons(test_predictions)


def blending_ensemble(
    holdout_predictions: list[pd.DataFrame],
    holdout_target: pd.DataFrame,
    test_predictions: list[pd.DataFrame],
    **kwargs: Any,
) -> pd.DataFrame:
    """Functional wrapper: fit blending meta-learner, return test predictions."""
    blend = BlendingEnsemble(**kwargs).fit(holdout_predictions, holdout_target)
    return blend.predict_proba_horizons(test_predictions)
