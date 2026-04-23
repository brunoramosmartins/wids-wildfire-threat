"""Accelerated Failure Time (AFT) models via lifelines (Phase 6.5).

Adds parametric survival baselines — Weibull AFT and LogNormal AFT —
as complementary members of the ensemble. When the data-generating
distribution is close to the assumed form, AFT models are extremely
efficient in small samples (n=221) and provide diversity against the
non-parametric boosters (GBS, RSF) and classifier-per-horizon (XGB).

Exposes the same ``fit`` + ``predict_proba_horizons`` interface as the
rest of the Phase 4–6 pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter, WeibullAFTFitter
from sklearn.base import BaseEstimator

from src.models.baselines import HORIZONS

__all__ = [
    "LogNormalAFTHorizonModel",
    "WeibullAFTHorizonModel",
    "get_lognormal_aft_model",
    "get_weibull_aft_model",
]


class _AFTHorizonBase(BaseEstimator):
    """Shared fit/predict logic for Weibull & LogNormal AFT."""

    FITTER_CLS: type[Any]

    def __init__(self, penalizer: float = 0.01, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fitter_: Any = None
        self._feature_cols: list[str] = []
        self._train_medians_: pd.Series | None = None

    def _prep_train(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        num = X.select_dtypes(include=[np.number]).copy()
        # AFT requires strictly positive times
        T = np.clip(np.asarray(y["time_to_hit_hours"].values, dtype=float), 1e-3, None)
        E = np.asarray(y["event"].values, dtype=int)
        self._feature_cols = num.columns.tolist()
        self._train_medians_ = num.median()
        num = num.fillna(self._train_medians_)
        num["time_to_hit_hours"] = T
        num["event"] = E
        return num

    def _prep_predict(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._feature_cols[: X.shape[1]])
        num = X[self._feature_cols].copy()
        if self._train_medians_ is not None:
            num = num.fillna(self._train_medians_)
        return num.astype(np.float64)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame,
        **kwargs: Any,
    ) -> _AFTHorizonBase:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = self._prep_train(X, y)
        self.fitter_ = self.FITTER_CLS(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.fitter_.fit(df, duration_col="time_to_hit_hours", event_col="event")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba_horizons(X)["prob_72h"].values
        result: np.ndarray = (probs >= 0.5).astype(np.int32)
        return result

    def predict_proba_horizons(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        Xdf = self._prep_predict(X)
        times = np.array(HORIZONS, dtype=float)
        surv = self.fitter_.predict_survival_function(Xdf, times=times)
        # lifelines returns (n_times, n_samples) normally; we want (n_samples, n_times)
        n = len(Xdf)
        if surv.shape[0] != n and surv.shape[1] == n:
            surv = surv.T
        elif surv.shape[0] != n:
            raise ValueError(f"AFT survival shape {surv.shape} incompatible with n={n}")
        out: dict[str, np.ndarray] = {}
        for h in HORIZONS:
            col = float(h)
            if col in surv.columns:
                s = surv[col].values.astype(float)
            else:
                nearest = min(surv.columns, key=lambda c: abs(float(c) - col))
                s = surv[nearest].values.astype(float)
            out[f"prob_{h}h"] = np.clip(1.0 - s, 0.0, 1.0)
        result = pd.DataFrame(out, index=Xdf.index)
        for i in range(1, len(HORIZONS)):
            prev_c = f"prob_{HORIZONS[i - 1]}h"
            curr_c = f"prob_{HORIZONS[i]}h"
            result[curr_c] = np.maximum(result[curr_c].values, result[prev_c].values)
        return result


class WeibullAFTHorizonModel(_AFTHorizonBase):
    """Weibull AFT on numeric features."""

    FITTER_CLS = WeibullAFTFitter


class LogNormalAFTHorizonModel(_AFTHorizonBase):
    """Log-Normal AFT on numeric features."""

    FITTER_CLS = LogNormalAFTFitter


def get_weibull_aft_model(
    penalizer: float = 0.01,
    l1_ratio: float = 0.0,
) -> WeibullAFTHorizonModel:
    return WeibullAFTHorizonModel(penalizer=penalizer, l1_ratio=l1_ratio)


def get_lognormal_aft_model(
    penalizer: float = 0.01,
    l1_ratio: float = 0.0,
) -> LogNormalAFTHorizonModel:
    return LogNormalAFTHorizonModel(penalizer=penalizer, l1_ratio=l1_ratio)
