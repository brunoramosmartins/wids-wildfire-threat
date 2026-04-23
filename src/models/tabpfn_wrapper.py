"""TabPFN wrapper for multi-horizon threat probabilities (Phase 6.5).

TabPFN is a foundation model for small tabular data (<= ~10,000 samples,
500 features) that is frequently the strongest single model in Kaggle
competitions with n < 1000. It does zero tuning and generalizes from a
fixed pre-training on synthetic tabular datasets.

We use it per-horizon (binary classification: "hit by H?") inside the
existing ``MultiHorizonClassifier`` interface. The ``tabpfn`` package is
optional — install via ``pip install -e ".[advanced]"``. This module
raises ``ImportError`` with a helpful message if it is missing.

Trade-off: inference time on CPU scales as O(n_train^2 * n_test). With
n_train=221 and n_test=95 this is negligible. On a GPU TabPFN is faster
but not required for this dataset size.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines import HORIZONS, MultiHorizonClassifier
from src.models.ensemble import enforce_monotonicity

__all__ = [
    "TabPFNHorizonModel",
    "get_tabpfn_model",
]

_TABPFN_IMPORT_ERROR_MSG = (
    "tabpfn is not installed. Install the optional extra:\n"
    '    pip install -e ".[advanced]"\n'
    "(this pulls PyTorch ~500MB)."
)


def _import_tabpfn() -> Any:
    """Import TabPFNClassifier lazily. Return type is Any because the
    real type is only known when the optional [advanced] extra is installed.
    """
    try:
        from tabpfn import TabPFNClassifier  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(_TABPFN_IMPORT_ERROR_MSG) from e
    return TabPFNClassifier


class TabPFNHorizonModel:
    """Per-horizon TabPFN wrapper exposing ``fit`` + ``predict_proba_horizons``.

    TabPFN does not accept sklearn Pipeline wrapping trivially because
    of its device/fit-time contract, so we manage the 4 per-horizon models
    directly here (mirrors ``MultiHorizonClassifier`` structure).

    Parameters
    ----------
    device:
        "cpu" or "cuda". Defaults to "cpu" for portability.
    n_estimators:
        Number of ensemble members inside TabPFN v2 (default 4).
    random_state:
        Reproducibility seed.
    """

    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = 4,
        random_state: int = 42,
    ) -> None:
        self.device = device
        self.n_estimators = n_estimators
        self.random_state = random_state
        # Value is TabPFNClassifier (or None when a horizon has no positives);
        # typed as Any to avoid optional-dependency typing issues.
        self.models_: dict[int, Any] = {}
        self._feature_cols: list[str] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame,
    ) -> TabPFNHorizonModel:
        TabPFNClassifier = _import_tabpfn()
        if isinstance(X, pd.DataFrame):
            self._feature_cols = X.columns.tolist()
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        T = np.asarray(y["time_to_hit_hours"].values, dtype=float)
        E = np.asarray(y["event"].values, dtype=int)

        self.models_ = {}
        for h in HORIZONS:
            label = ((E == 1) & (T <= h)).astype(np.int32)
            # TabPFN requires both classes present
            if len(np.unique(label)) < 2:
                self.models_[h] = None
                continue
            clf = TabPFNClassifier(
                device=self.device,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            clf.fit(X_np, label)
            self.models_[h] = clf
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba_horizons(X)["prob_72h"].values
        result: np.ndarray = (probs >= 0.5).astype(np.int32)
        return result

    def predict_proba_horizons(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
            idx = X.index
        else:
            X_np = np.asarray(X, dtype=np.float32)
            idx = pd.RangeIndex(len(X_np))

        out: dict[str, np.ndarray] = {}
        for h in HORIZONS:
            model = self.models_.get(h)
            if model is None:
                # Fallback: marginal rate (0.0 if no positives seen)
                out[f"prob_{h}h"] = np.zeros(len(X_np), dtype=float)
            else:
                p = model.predict_proba(X_np)[:, 1]
                out[f"prob_{h}h"] = p
        result = pd.DataFrame(out, index=idx)
        return enforce_monotonicity(result)


def get_tabpfn_model(
    device: str = "cpu",
    n_estimators: int = 4,
    random_state: int = 42,
) -> TabPFNHorizonModel:
    """Return a fresh TabPFN multi-horizon classifier."""
    return TabPFNHorizonModel(
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
    )


# Keep the reference so linters don't complain when advanced extra is absent
_ = MultiHorizonClassifier
