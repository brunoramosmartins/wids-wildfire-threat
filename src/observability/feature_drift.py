"""Feature drift detection.

Compares feature distributions between train and test sets
using PSI and chi-square tests.
"""

from __future__ import annotations

import pandas as pd


def compute_psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    raise NotImplementedError
