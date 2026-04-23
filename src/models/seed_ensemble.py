"""Seed-ensembling wrapper (Phase 6.5).

Fits the same model factory with multiple random seeds and averages
their per-horizon probability predictions. Reduces stochastic variance
from random-feature sampling (XGB/LGBM), bootstrap draws (RSF), and
gradient boosting shuffle order.

In small-sample regimes (n=221) this is especially valuable: a single
seed's predictions can shift 1–2 pp between seeds, which is material
against the Weighted Brier component (70% of Hybrid Score).

Usage
-----
    from src.models.seed_ensemble import SeedEnsembleWrapper
    from src.models.boosting import get_xgboost_model

    def factory():
        return get_xgboost_model(n_estimators=300, max_depth=4)

    model = SeedEnsembleWrapper(factory, seeds=[42, 123, 2024, 7, 99])
    model.fit(X_train, y_train)
    preds = model.predict_proba_horizons(X_test)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.models.baselines import HORIZONS
from src.models.ensemble import enforce_monotonicity

PROB_COLS = [f"prob_{h}h" for h in HORIZONS]


class SeedEnsembleWrapper(BaseEstimator):
    """Average-of-seeds wrapper that exposes ``predict_proba_horizons``.

    Parameters
    ----------
    factory:
        Zero-arg callable returning a fresh model instance. Must support
        ``fit(X, y)`` and ``predict_proba_horizons(X) -> DataFrame``.
        The factory should NOT hardcode the seed inside; prefer passing
        ``random_state`` explicitly if the factory accepts it.
    seeds:
        Iterable of integer seeds. The factory is called once per seed;
        the wrapper monkey-patches a ``random_state`` attribute on the
        returned estimator if one exists.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        seeds: list[int] | tuple[int, ...] = (42, 123, 2024, 7, 99),
    ) -> None:
        self.factory = factory
        self.seeds = list(seeds)
        self.models_: list[Any] = []

    def _set_seed(self, model: Any, seed: int) -> None:
        """Best-effort: set random_state on common sklearn-like estimators."""
        if hasattr(model, "random_state"):
            try:
                model.random_state = seed
            except AttributeError:
                pass
        # Nested sklearn Pipeline: set on final estimator
        if hasattr(model, "estimator") and hasattr(model.estimator, "random_state"):
            try:
                model.estimator.random_state = seed
            except AttributeError:
                pass
        # Phase 4 MultiHorizonClassifier wraps a base_estimator
        if hasattr(model, "base_estimator") and hasattr(model.base_estimator, "random_state"):
            try:
                model.base_estimator.random_state = seed
            except AttributeError:
                pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs: Any) -> SeedEnsembleWrapper:
        self.models_ = []
        for s in self.seeds:
            model = self.factory()
            self._set_seed(model, int(s))
            model.fit(X, y)
            self.models_.append(model)
        return self

    def predict_proba_horizons(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("SeedEnsembleWrapper: call fit() before predict_proba_horizons()")
        # Collect (n_samples, n_horizons) arrays
        preds = [m.predict_proba_horizons(X)[PROB_COLS].values.astype(float) for m in self.models_]
        avg = np.mean(preds, axis=0)
        n = avg.shape[0]
        # Preserve index if X is a DataFrame
        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(n)
        out = pd.DataFrame(avg, columns=PROB_COLS, index=idx)
        return enforce_monotonicity(out)
