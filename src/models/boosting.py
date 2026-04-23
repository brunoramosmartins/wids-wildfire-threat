"""Gradient boosting models for multi-horizon threat probabilities.

Uses one binary classifier per horizon (same framing as baselines) with
XGBoost, LightGBM, or CatBoost. Handles missing values where the library
supports it (CatBoost / XGBoost / recent LightGBM).

Monotone constraints (Phase 6.5): sign of the expected relationship between
feature and P(hit by T). +1 means "increase feature -> increase risk", -1
means "increase feature -> decrease risk". Unknown features get 0 (no
constraint). These are domain priors from the data dictionary, not learned.
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from src.models.baselines import MultiHorizonClassifier

__all__ = [
    "MONOTONE_CONSTRAINTS_DEFAULT",
    "get_catboost_model",
    "get_lightgbm_model",
    "get_sklearn_gbdt_model",
    "get_xgboost_model",
    "monotone_vector",
]


# Feature → monotone sign. Known physical priors from docs/data_dictionary.md.
# +1 = higher feature ⇒ higher P(hit); -1 = higher feature ⇒ lower P(hit).
MONOTONE_CONSTRAINTS_DEFAULT: dict[str, int] = {
    # Distance: farther = safer
    "dist_min_ci_0_5h": -1,
    "dist_min_log": -1,
    # Closing: higher = more dangerous
    "closing_speed_m_per_h": +1,
    "closing_speed_abs_m_per_h": +1,  # dropped in features but safe default
    "projected_advance_m": +1,
    "advance_ratio": +1,
    "threat_score": +1,
    "alignment_x_speed": +1,
    # Directionality: more aligned = more dangerous
    "alignment_abs": +1,
    # Growth: larger/faster growing = more dangerous
    "area_first_ha": +1,
    "log1p_area_first": +1,
    "area_growth_rate_ha_per_h": +1,
    "radial_growth_m": +1,
    "growth_x_proximity": +1,
    "speed_x_growth": +1,
    # Binary flags
    "is_closing": +1,
    "is_growing": +1,
    # Inverse (distance-away movement): positive means moving away → safer
    "dist_change_ci_0_5h": -1,
    "dist_slope_ci_0_5h": -1,
    # Projected arrival time: longer time = safer
    "projected_arrival_h": -1,
}


def monotone_vector(
    feature_names: list[str],
    overrides: dict[str, int] | None = None,
) -> list[int]:
    """Build the per-feature monotone-constraint vector for a feature list."""
    base = dict(MONOTONE_CONSTRAINTS_DEFAULT)
    if overrides:
        base.update(overrides)
    return [int(base.get(f, 0)) for f in feature_names]


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
    monotone_constraints: list[int] | tuple[int, ...] | str | None = None,
    feature_names: list[str] | None = None,
) -> MultiHorizonClassifier:
    """Binary XGBoost per horizon (prob. of hit-by-T).

    If ``monotone_constraints`` is None and ``feature_names`` is provided,
    constraints are auto-derived from ``MONOTONE_CONSTRAINTS_DEFAULT``.
    Pass ``monotone_constraints=[]`` explicitly to disable.
    """
    mc = monotone_constraints
    if mc is None and feature_names is not None:
        mc = monotone_vector(feature_names)
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
        monotone_constraints=tuple(mc) if isinstance(mc, list) else mc,
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
    monotone_constraints: list[int] | None = None,
    feature_names: list[str] | None = None,
) -> MultiHorizonClassifier:
    """Binary LightGBM per horizon.

    See ``get_xgboost_model`` for monotone-constraint behavior.
    """
    mc = monotone_constraints
    if mc is None and feature_names is not None:
        mc = monotone_vector(feature_names)
    kwargs: dict = {}
    if mc is not None:
        kwargs["monotone_constraints"] = list(mc)
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
        **kwargs,
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
