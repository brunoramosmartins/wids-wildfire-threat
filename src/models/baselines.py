"""Baseline model implementations.

Simple models to establish performance floor:
Logistic Regression, Random Forest.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator


def get_logistic_baseline() -> BaseEstimator:
    """Create a Logistic Regression baseline model."""
    raise NotImplementedError


def get_random_forest_baseline() -> BaseEstimator:
    """Create a Random Forest baseline model."""
    raise NotImplementedError
