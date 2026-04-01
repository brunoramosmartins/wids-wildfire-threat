"""Survival analysis model implementations.

Cox Proportional Hazards, Random Survival Forest,
and Gradient Boosted Survival models.
"""

from __future__ import annotations

from typing import Any


def get_cox_ph_model() -> Any:
    """Create a Cox Proportional Hazards model (lifelines)."""
    raise NotImplementedError


def get_rsf_model() -> Any:
    """Create a Random Survival Forest model (scikit-survival)."""
    raise NotImplementedError


def get_gbs_model() -> Any:
    """Create a Gradient Boosted Survival model (scikit-survival)."""
    raise NotImplementedError
