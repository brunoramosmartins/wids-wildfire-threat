"""Feature selection utilities.

Mutual information scoring, permutation importance, and
correlation-based filtering for feature ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def rank_mutual_information(
    df: pd.DataFrame, target: pd.Series, random_state: int = 42
) -> pd.Series:
    """Rank features by mutual information with target.

    Returns a Series of MI scores sorted descending.
    """
    mi = mutual_info_regression(df, target, random_state=random_state)
    scores = pd.Series(mi, index=df.columns, name="mi_score")
    return scores.sort_values(ascending=False)


def find_correlated_pairs(
    df: pd.DataFrame, threshold: float = 0.95
) -> list[tuple[str, str, float]]:
    """Find feature pairs with absolute correlation above threshold."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val > threshold:
                pairs.append((idx, col, round(float(val), 4)))
    return sorted(pairs, key=lambda x: -x[2])


def select_features(df: pd.DataFrame, target: pd.Series) -> list[str]:
    """Select top features using MI ranking and correlation filtering.

    Returns list of selected feature names.
    """
    mi_scores = rank_mutual_information(df, target)

    # Drop features with near-zero MI
    keep = mi_scores[mi_scores > 0.01].index.tolist()

    # Among highly correlated pairs, drop the one with lower MI
    corr_pairs = find_correlated_pairs(df[keep])
    to_drop: set[str] = set()
    for f1, f2, _ in corr_pairs:
        if f1 in to_drop or f2 in to_drop:
            continue
        score1 = mi_scores.get(f1, 0)
        score2 = mi_scores.get(f2, 0)
        to_drop.add(f2 if score1 >= score2 else f1)

    return [f for f in keep if f not in to_drop]
