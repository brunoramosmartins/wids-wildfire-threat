"""Gradient boosting models for multi-horizon threat probabilities.

Uses one binary classifier per horizon (same framing as baselines) with
XGBoost, LightGBM, or CatBoost. Handles missing values where the library
supports it (CatBoost / XGBoost / recent LightGBM).
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from src.models.baselines import MultiHorizonClassifier

__all__ = [
    "get_catboost_model",
    "get_lightgbm_model",
    "get_sklearn_gbdt_model",
    "get_xgboost_model",
]


def get_xgboost_model(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> MultiHorizonClassifier:
    """Binary XGBoost per horizon (prob. of hit-by-T)."""
    base = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="binary:logistic",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        eval_metric="logloss",
        enable_categorical=False,
    )
    return MultiHorizonClassifier(base_estimator=base)


def get_lightgbm_model(
    n_estimators: int = 300,
    max_depth: int = -1,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 0.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> MultiHorizonClassifier:
    """Binary LightGBM per horizon."""
    base = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="binary",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )
    return MultiHorizonClassifier(base_estimator=base)


def get_catboost_model(
    iterations: int = 300,
    depth: int = 6,
    learning_rate: float = 0.1,
    l2_leaf_reg: float = 3.0,
    random_state: int = 42,
    loss_function: str = "Logloss",
) -> MultiHorizonClassifier:
    """Binary CatBoost per horizon; robust to NaNs in features."""
    base = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_state=random_state,
        loss_function=loss_function,
        verbose=False,
        allow_writing_files=False,
    )
    return MultiHorizonClassifier(base_estimator=base)


def get_sklearn_gbdt_model(
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    random_state: int = 42,
) -> MultiHorizonClassifier:
    """Sklearn histogram GBDT fallback if GPU / libs differ across machines."""
    base = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
    )
    return MultiHorizonClassifier(base_estimator=base)
