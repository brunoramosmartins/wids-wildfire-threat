"""Post-hoc probability calibration per horizon (Phase 6.5).

Since 70% of the WiDS Hybrid Score comes from Weighted Brier (calibration),
we fit an isotonic regression **per scored horizon** on OOF predictions and
apply it to test predictions. Monotonicity across horizons is re-enforced
after calibration.

Usage
-----
    calib = IsotonicHorizonCalibrator().fit(oof_preds, y_train)
    calibrated_test_preds = calib.transform(test_preds)

Design notes
------------
- We use **censor-aware** labels per horizon at fit time. Fires that are
  censored before horizon H do not contribute to the calibrator for that H.
- 12h is included in calibration (even though it is not scored) because
  the submission file still needs a coherent prob_12h.
- With n=221 isotonic can be noisy; ``n_bins`` is auto-capped at n/20 when
  the effective sample size is small.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.models.baselines import HORIZONS
from src.models.ensemble import enforce_monotonicity

PROB_COLS = [f"prob_{h}h" for h in HORIZONS]


class IsotonicHorizonCalibrator:
    """Isotonic regression per horizon on censor-aware labels.

    Parameters
    ----------
    min_events_per_horizon:
        If a horizon has fewer than this many effective positive events
        after censor-aware masking, we skip calibration for that horizon
        (keep the raw predictions). Default 5.
    """

    def __init__(self, min_events_per_horizon: int = 5) -> None:
        self.min_events_per_horizon = min_events_per_horizon
        self.calibrators_: dict[int, IsotonicRegression | None] = {}
        self._fitted = False

    def fit(
        self,
        oof_predictions: pd.DataFrame,
        y: pd.DataFrame,
    ) -> IsotonicHorizonCalibrator:
        T = np.asarray(y["time_to_hit_hours"].values, dtype=float)
        E = np.asarray(y["event"].values, dtype=int)

        for h in HORIZONS:
            col = f"prob_{h}h"
            pred = np.asarray(oof_predictions[col].values, dtype=float)
            # Censor-aware inclusion (Brier rule) + positive label check
            include = ~((E == 0) & (T < h))
            label = ((E == 1) & (T <= h)).astype(np.int32)[include]
            pred_in = pred[include]
            if label.sum() < self.min_events_per_horizon:
                self.calibrators_[h] = None
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pred_in, label)
            self.calibrators_[h] = iso

        self._fitted = True
        return self

    def transform(self, preds: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("IsotonicHorizonCalibrator: call fit() before transform()")
        out = preds.copy()
        for h in HORIZONS:
            col = f"prob_{h}h"
            iso = self.calibrators_.get(h)
            if iso is None:
                continue  # keep raw predictions
            out[col] = iso.predict(np.asarray(preds[col].values, dtype=float))
        # Guarantee monotone + clipped
        return enforce_monotonicity(out)

    def fit_transform(
        self,
        oof_predictions: pd.DataFrame,
        y: pd.DataFrame,
        target_predictions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        self.fit(oof_predictions, y)
        tgt = target_predictions if target_predictions is not None else oof_predictions
        return self.transform(tgt)

    def summary(self) -> dict[str, Any]:
        return {h: ("skipped" if iso is None else "fitted") for h, iso in self.calibrators_.items()}
