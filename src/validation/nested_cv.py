"""Nested cross-validation helper for honest tuning (Phase 6.5).

Problem: Optuna-tuning and reporting performance on the **same** K-fold
split overestimates out-of-sample performance. The Optuna sampler adapts
to fold-specific noise.

Nested CV fixes this:
- Outer loop: K folds for reporting.
- Inner loop: K (or K-1) folds inside each outer train set for tuning.

Returned: per-outer-fold {best_params, held-out Hybrid Score}. The
mean of held-out scores is the honest performance estimate.

Usage (see `src/models/tune.py` for the inner-loop objective machinery):

    from src.validation.nested_cv import nested_cv_tune
    result = nested_cv_tune(
        model_name="gradient_boosted_survival",
        search_space=...,
        X=X, y=y,
        n_outer=5, n_inner=5,
        n_trials=30,
    )

NOTE: this is compute-expensive (~5× the cost of regular tuning). Use
only as a final sanity check before selecting the best model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold

from src.models.evaluate import compute_metrics
from src.models.tune import _FACTORIES, _SUGGESTERS
from src.observability.logger import setup_logger

logger = setup_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _inner_hybrid(
    params: dict[str, Any],
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_inner: int,
    seed: int,
) -> float:
    """Mean Hybrid across n_inner folds for one parameter set."""
    factory_fn = _FACTORIES[model_name]
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    scores: list[float] = []
    for tr, va in skf.split(X, y["event"]):
        model = factory_fn(**params)
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict_proba_horizons(X.iloc[va])
        scores.append(compute_metrics(y.iloc[va], preds)["hybrid_score"])
    return float(np.mean(scores))


def nested_cv_tune(
    model_name: str,
    search_space: dict[str, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_outer: int = 5,
    n_inner: int = 5,
    n_trials: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Nested CV for a single model. Returns held-out Hybrid per outer fold."""
    if model_name not in _SUGGESTERS:
        raise ValueError(f"Unknown model for nested tuning: {model_name}")

    suggest = _SUGGESTERS[model_name]
    factory_fn = _FACTORIES[model_name]

    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    per_fold: list[dict[str, Any]] = []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y["event"]), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        def objective(trial: optuna.Trial, X_tr=X_tr, y_tr=y_tr) -> float:
            params = suggest(trial, search_space)
            params["random_state"] = seed
            try:
                return _inner_hybrid(params, model_name, X_tr, y_tr, n_inner, seed)
            except Exception:
                raise optuna.TrialPruned() from None

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))
        best_params = study.best_params
        best_params["random_state"] = seed

        # Refit on full outer train, evaluate on held-out outer fold
        model = factory_fn(**best_params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba_horizons(X_te)
        held_out = compute_metrics(y_te, preds)["hybrid_score"]

        logger.info(
            "nested_cv_fold",
            model=model_name,
            fold=fold,
            held_out_hybrid=round(held_out, 5),
            inner_best_hybrid=round(study.best_value, 5),
        )
        per_fold.append(
            {
                "fold": fold,
                "held_out_hybrid": held_out,
                "inner_best_hybrid": study.best_value,
                "best_params": best_params,
            }
        )

    held_out_scores = [r["held_out_hybrid"] for r in per_fold]
    return {
        "model": model_name,
        "outer_hybrid_mean": float(np.mean(held_out_scores)),
        "outer_hybrid_std": float(np.std(held_out_scores)),
        "n_outer": n_outer,
        "n_inner": n_inner,
        "n_trials_per_inner": n_trials,
        "per_fold": per_fold,
    }
