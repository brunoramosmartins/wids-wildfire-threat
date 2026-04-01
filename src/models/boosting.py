"""Gradient boosting model implementations.

XGBoost and LightGBM with appropriate objective functions.
"""

from __future__ import annotations

from typing import Any


def get_xgboost_model() -> Any:
    """Create an XGBoost model."""
    raise NotImplementedError


def get_lightgbm_model() -> Any:
    """Create a LightGBM model."""
    raise NotImplementedError
