"""Survival analysis model implementations.

Cox Proportional Hazards (lifelines), Random Survival Forest,
and Gradient Boosted Survival (scikit-survival).

Each wrapper exposes ``fit`` + ``predict_proba_horizons`` so training
and Kaggle submission stay compatible with the baseline pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

from src.models.baselines import HORIZONS

__all__ = [
    "CoxPHHorizonModel",
    "GradientBoostedSurvivalHorizonModel",
    "RandomSurvivalForestHorizonModel",
    "get_cox_ph_model",
    "get_gbs_model",
    "get_rsf_model",
]


def _y_to_sksurv(y: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            (bool(ev), float(t))
            for ev, t in zip(y["event"].values, y["time_to_hit_hours"].values, strict=True)
        ],
        dtype=[("event", bool), ("time", float)],
    )


class CoxPHHorizonModel(BaseEstimator):
    """Cox PH on numeric features; horizon probs from ``1 - S(t|x)``."""

    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.cph_ = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self._feature_cols: list[str] = []
        self._train_medians_: pd.Series | None = None

    def _prepare_train_frame(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        num = X.select_dtypes(include=[np.number]).copy()
        self._feature_cols = num.columns.tolist()
        self._train_medians_ = num.median()
        num = num.fillna(self._train_medians_)
        num["time_to_hit_hours"] = y["time_to_hit_hours"].values
        num["event"] = y["event"].values
        return num

    def _prepare_predict_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            cols = self._feature_cols[: X.shape[1]] if self._feature_cols else None
            if cols and len(cols) == X.shape[1]:
                X = pd.DataFrame(X, columns=cols)
            else:
                X = pd.DataFrame(X)
        assert self._feature_cols
        num = X[self._feature_cols].copy()
        if self._train_medians_ is not None:
            num = num.fillna(self._train_medians_)
        else:
            num = num.fillna(num.median())
        return num.astype(np.float64)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame,
        **kwargs: Any,
    ) -> CoxPHHorizonModel:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = self._prepare_train_frame(X, y)
        self.cph_.fit(df, duration_col="time_to_hit_hours", event_col="event")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba_horizons(X)["prob_72h"].values
        return (probs >= 0.5).astype(np.int32)

    def predict_proba_horizons(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        Xdf = self._prepare_predict_frame(X)
        times = np.array(HORIZONS, dtype=float)
        surv = self.cph_.predict_survival_function(Xdf, times=times)
        # lifelines often returns (n_times, n_samples); training expects (n_samples, n_times).
        n = len(Xdf)
        if surv.shape[0] != n and surv.shape[1] == n:
            surv = surv.T
        elif surv.shape[0] != n:
            raise ValueError(f"Cox survival shape {surv.shape} incompatible with n_samples={n}")
        out: dict[str, np.ndarray] = {}
        for h in HORIZONS:
            col = float(h)
            if col in surv.columns:
                s = surv[col].values.astype(float)
            else:
                nearest = min(surv.columns, key=lambda c: abs(float(c) - col))
                s = surv[nearest].values.astype(float)
            out[f"prob_{h}h"] = np.clip(1.0 - s, 0.0, 1.0)
        result = pd.DataFrame(out)
        for i in range(1, len(HORIZONS)):
            prev_c = f"prob_{HORIZONS[i - 1]}h"
            curr_c = f"prob_{HORIZONS[i]}h"
            result[curr_c] = np.maximum(result[curr_c].values, result[prev_c].values)
        return result


def _survival_fn_values(pred_fns: np.ndarray, t: float) -> np.ndarray:
    """Evaluate scikit-survival survival step functions at ``t`` (clamped to fit domain)."""
    vals = np.empty(len(pred_fns), dtype=float)
    for i, fn in enumerate(pred_fns):
        lo, hi = float(fn.domain[0]), float(fn.domain[1])
        t_use = float(np.clip(t, lo, hi))
        vals[i] = float(fn(t_use))
    return vals


class _SkSurvEnsembleHorizonModel(BaseEstimator):
    """Shared logic for RSF / gradient boosting survival."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self._feature_cols: list[str] = []
        self._train_medians_: pd.Series | None = None

    def _as_float_matrix(self, X: pd.DataFrame | np.ndarray, fit: bool) -> np.ndarray:
        if isinstance(X, np.ndarray):
            return X.astype(np.float64)
        num = X.select_dtypes(include=[np.number]).copy()
        if fit:
            self._feature_cols = num.columns.tolist()
            self._train_medians_ = num.median()
        assert self._feature_cols
        num = num[self._feature_cols].fillna(self._train_medians_ if fit else self._train_medians_)
        if not fit:
            num = num.fillna(num.median())
        return num.astype(np.float64).values

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame,
        **kwargs: Any,
    ) -> _SkSurvEnsembleHorizonModel:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_arr = self._as_float_matrix(X, fit=True)
        y_arr = _y_to_sksurv(y)
        self.estimator.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba_horizons(X)["prob_72h"].values
        return (probs >= 0.5).astype(np.int32)

    def predict_proba_horizons(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X_arr = X.astype(np.float64)
        else:
            X_arr = self._as_float_matrix(X, fit=False)
        fns = self.estimator.predict_survival_function(X_arr)
        out: dict[str, np.ndarray] = {}
        for h in HORIZONS:
            s = _survival_fn_values(fns, float(h))
            out[f"prob_{h}h"] = np.clip(1.0 - s, 0.0, 1.0)
        result = pd.DataFrame(out)
        for i in range(1, len(HORIZONS)):
            prev_c = f"prob_{HORIZONS[i - 1]}h"
            curr_c = f"prob_{HORIZONS[i]}h"
            result[curr_c] = np.maximum(result[curr_c].values, result[prev_c].values)
        return result


class RandomSurvivalForestHorizonModel(_SkSurvEnsembleHorizonModel):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        super().__init__(rsf)


class GradientBoostedSurvivalHorizonModel(_SkSurvEnsembleHorizonModel):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int = 42,
    ) -> None:
        gbs = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            dropout_rate=0.0,
        )
        super().__init__(gbs)


def get_cox_ph_model(
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
) -> CoxPHHorizonModel:
    return CoxPHHorizonModel(penalizer=penalizer, l1_ratio=l1_ratio)


def get_rsf_model(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 6,
    min_samples_leaf: int = 3,
    random_state: int = 42,
) -> RandomSurvivalForestHorizonModel:
    return RandomSurvivalForestHorizonModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


def get_gbs_model(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> GradientBoostedSurvivalHorizonModel:
    return GradientBoostedSurvivalHorizonModel(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
