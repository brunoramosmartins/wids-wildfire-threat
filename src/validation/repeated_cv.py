"""Repeated stratified K-fold CV wrapper (Phase 6.5).

Runs StratifiedKFold ``n_repeats`` times with different seeds and
aggregates Hybrid Score / Weighted Brier / C-index statistics.

Motivation: n=221 with ~31% event rate means each 5-fold CV has very few
positives per fold (~14). Variance across seeds is material. 5 folds × 10
seeds = 50 evaluations, ~√10 variance reduction vs single-seed K-fold.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.evaluate import compute_metrics
from src.observability.logger import setup_logger

logger = setup_logger(__name__)


def repeated_stratified_cv(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 10,
    base_seed: int = 42,
    log_each_repeat: bool = False,
) -> dict[str, Any]:
    """Repeated stratified K-fold CV returning summary statistics.

    Each repeat uses ``base_seed + r`` as the fold seed. Within each repeat
    we iterate K folds and compute per-fold Hybrid Score + components.

    Returns dict with:
    - ``hybrid_score_mean``, ``hybrid_score_std``  (across all fold×repeat evaluations)
    - ``weighted_brier_mean``, ``weighted_brier_std``
    - ``c_index_mean``, ``c_index_std``
    - ``fold_scores``: flat list of per-fold Hybrid scores
    - ``n_evaluations``: n_splits × n_repeats
    """
    fold_hybrid: list[float] = []
    fold_wbrier: list[float] = []
    fold_cindex: list[float] = []

    for r in range(n_repeats):
        seed = base_seed + r
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        repeat_scores: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y["event"]), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model = model_factory()
            model.fit(X_tr, y_tr)
            preds = model.predict_proba_horizons(X_va)
            m = compute_metrics(y_va, preds)
            fold_hybrid.append(m["hybrid_score"])
            fold_wbrier.append(m["weighted_brier"])
            fold_cindex.append(m["c_index"])
            repeat_scores.append(m["hybrid_score"])
        if log_each_repeat:
            logger.info(
                "repeated_cv_repeat",
                repeat=r + 1,
                seed=seed,
                hybrid_mean=round(float(np.mean(repeat_scores)), 5),
            )

    return {
        "hybrid_score_mean": float(np.mean(fold_hybrid)),
        "hybrid_score_std": float(np.std(fold_hybrid)),
        "weighted_brier_mean": float(np.mean(fold_wbrier)),
        "weighted_brier_std": float(np.std(fold_wbrier)),
        "c_index_mean": float(np.mean(fold_cindex)),
        "c_index_std": float(np.std(fold_cindex)),
        "fold_scores": fold_hybrid,
        "n_evaluations": len(fold_hybrid),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
    }
