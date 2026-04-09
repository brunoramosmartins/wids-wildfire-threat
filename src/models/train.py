"""Training orchestrator.

Loads feature data, applies stratified K-fold cross-validation,
trains baseline models, evaluates, and logs results.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.baselines import (
    get_kaplan_meier_baseline,
    get_logistic_baseline,
    get_random_forest_baseline,
)
from src.models.evaluate import compute_metrics, evaluate_model
from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


class HorizonPredictor(Protocol):
    """Models used in train.py expose fit + per-horizon probabilities."""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs: Any) -> Any: ...

    def predict_proba_horizons(self, X: pd.DataFrame) -> pd.DataFrame: ...


def _get_feature_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split feature matrix into X, y, and feature list."""
    config = load_config(Path("configs/data_config.yaml"))
    target_col = config["target_column"]
    event_col = config["event_column"]
    id_col = config["id_column"]

    meta_cols = [id_col, target_col, event_col]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols]
    y = df[[target_col, event_col]]
    return X, y, feature_cols


def _cross_validate(
    model_factory: Callable[[], HorizonPredictor],
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Run stratified K-fold CV and return aggregate metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y["event"]), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict_proba_horizons(X_val)

        metrics = compute_metrics(y_val, preds)
        fold_metrics.append(metrics)

        logger.info(
            "fold_complete",
            model=model_name,
            fold=fold,
            brier_mean=round(metrics["brier_mean"], 4),
        )

    # Aggregate fold metrics
    agg: dict[str, float] = {}
    all_keys = fold_metrics[0].keys()
    for key in all_keys:
        values = [m[key] for m in fold_metrics if key in m]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))

    logger.info(
        "cv_complete",
        model=model_name,
        brier_mean=round(agg["brier_mean_mean"], 4),
        brier_std=round(agg["brier_mean_std"], 4),
    )

    return agg


def train_model() -> None:
    """Train all baseline models with cross-validation and log to MLflow."""
    config = load_config(Path("configs/model_config.yaml"))
    seed = config["random_seed"]
    n_splits = config["validation"]["n_splits"]
    set_seed(seed)

    # Load features
    data_config = load_config(Path("configs/data_config.yaml"))
    feat_path = Path(data_config["paths"]["features"]) / data_config["feature_files"]["train"]
    df = read_parquet(feat_path)
    X, y, feature_cols = _get_feature_and_target(df)

    logger.info(
        "training_start",
        samples=len(X),
        features=len(feature_cols),
        n_splits=n_splits,
    )

    # Define baseline models
    baselines: dict[str, tuple[Callable[[], HorizonPredictor], dict[str, Any]]] = {
        "kaplan_meier": (get_kaplan_meier_baseline, {"type": "kaplan_meier"}),
        "logistic_regression": (
            get_logistic_baseline,
            {"type": "logistic_regression", "C": 1.0},
        ),
        "random_forest": (
            get_random_forest_baseline,
            {"type": "random_forest", "n_estimators": 100},
        ),
    }

    results: dict[str, dict[str, float]] = {}

    for model_name, (factory, params) in baselines.items():
        logger.info("training_model", model=model_name)

        # Cross-validation
        cv_metrics = _cross_validate(factory, model_name, X, y, feature_cols, n_splits, seed)

        # Train final model on full training data
        final_model = factory()
        final_model.fit(X, y)
        full_preds = final_model.predict_proba_horizons(X)
        full_metrics = compute_metrics(y, full_preds)

        # Log to MLflow
        evaluate_model(
            model_name=model_name,
            y_true=y,
            y_pred=full_preds,
            params={**params, "n_splits": n_splits, "seed": seed},
            feature_list=feature_cols,
            model_artifact=final_model,
        )

        results[model_name] = {
            "brier_mean_cv": cv_metrics["brier_mean_mean"],
            "brier_std_cv": cv_metrics["brier_mean_std"],
            **full_metrics,
        }

    # Save results summary
    results_path = Path("models/baseline_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("results_saved", path=str(results_path))

    # Print summary
    for name, m in results.items():
        logger.info(
            "model_summary",
            model=name,
            brier_cv=f"{m['brier_mean_cv']:.4f} +/- {m['brier_std_cv']:.4f}",
            brier_full=f"{m['brier_mean']:.4f}",
        )


def main() -> None:
    """Run the training pipeline."""
    logger.info("Starting model training")
    train_model()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
