"""Baseline model implementations.

Simple models to establish performance floor:
- Kaplan-Meier (naive survival baseline)
- Random Forest Classifier (per-horizon binary classification)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = [12, 24, 48, 72]


class KaplanMeierBaseline(BaseEstimator, ClassifierMixin):
    """Naive survival baseline using overall Kaplan-Meier curve.

    Predicts the same P(hit by T) for every sample based on
    the training set's marginal survival function.
    """

    def __init__(self) -> None:
        self.kmf_ = KaplanMeierFitter()
        self.probs_: dict[int, float] = {}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame,
        **kwargs: Any,
    ) -> KaplanMeierBaseline:
        """Fit KM on training durations and events."""
        T = y["time_to_hit_hours"].values
        E = y["event"].values
        self.kmf_.fit(T, E)
        for h in HORIZONS:
            # P(hit by h) = 1 - S(h)
            self.probs_[h] = float(1 - self.kmf_.predict(h))
        return self

    def predict_proba_horizons(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return P(hit by T) for each horizon for all samples."""
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return pd.DataFrame({f"prob_{h}h": np.full(n, self.probs_[h]) for h in HORIZONS})


class MultiHorizonClassifier(BaseEstimator):
    """Trains independent binary classifiers for each time horizon.

    For each horizon T, creates label: 1 if event==1 AND time_to_hit <= T.
    """

    def __init__(self, base_estimator: BaseEstimator | None = None) -> None:
        self.base_estimator = base_estimator
        self.models_: dict[int, Pipeline] = {}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame,
        **kwargs: Any,
    ) -> MultiHorizonClassifier:
        """Fit one classifier per horizon."""
        T = y["time_to_hit_hours"].values
        E = y["event"].values

        for h in HORIZONS:
            # Label: hit within h hours
            labels = ((E == 1) & (T <= h)).astype(int)
            estimator = (
                self.base_estimator
                if self.base_estimator is not None
                else LogisticRegression(max_iter=1000)
            )
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", _clone_estimator(estimator)),
                ]
            )
            pipe.fit(X, labels)
            self.models_[h] = pipe

        return self

    def predict_proba_horizons(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return P(hit by T) for each horizon."""
        probs = {}
        for h in HORIZONS:
            p = self.models_[h].predict_proba(X)[:, 1]
            probs[f"prob_{h}h"] = p

        # Enforce monotonicity: P(hit by 12) <= P(hit by 24) <= ...
        result = pd.DataFrame(probs)
        for i in range(1, len(HORIZONS)):
            prev_col = f"prob_{HORIZONS[i - 1]}h"
            curr_col = f"prob_{HORIZONS[i]}h"
            result[curr_col] = np.maximum(result[curr_col].values, result[prev_col].values)
        return result


def _clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """Clone an estimator with the same parameters."""
    from sklearn.base import clone

    return clone(estimator)


def get_logistic_baseline() -> MultiHorizonClassifier:
    """Create a Logistic Regression baseline model."""
    return MultiHorizonClassifier(
        base_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    )


def get_random_forest_baseline() -> MultiHorizonClassifier:
    """Create a Random Forest baseline model."""
    return MultiHorizonClassifier(
        base_estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )


def get_kaplan_meier_baseline() -> KaplanMeierBaseline:
    """Create a Kaplan-Meier naive baseline."""
    return KaplanMeierBaseline()
