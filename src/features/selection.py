"""Feature selection utilities.

Mutual information scoring, permutation importance, RFE,
and top-N CV comparison for feature ranking.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.inspection import permutation_importance

from src.observability.logger import setup_logger

logger = setup_logger(__name__)


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


def hit_within_72_labels(y: pd.DataFrame) -> np.ndarray:
    """Binary label: observed hit within 72h (for surrogate ranking / RFE)."""
    return ((y["event"] == 1) & (y["time_to_hit_hours"] <= 72)).astype(np.int32).values


def recursive_feature_elimination_top(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_features_to_select: int = 30,
    step: int = 5,
    random_state: int = 42,
) -> list[str]:
    """RFE with RandomForest on 72h hit label; returns retained feature names."""
    labels = hit_within_72_labels(y)
    X_imp = X.fillna(X.median(numeric_only=True)).copy()
    for c in X_imp.select_dtypes(include=["object", "category"]).columns:
        X_imp[c] = pd.factorize(X_imp[c])[0]
    n_feat = X_imp.shape[1]
    if n_features_to_select >= n_feat:
        return X_imp.columns.tolist()
    step_use = max(1, min(step, n_feat - n_features_to_select))
    est = RandomForestClassifier(
        n_estimators=120,
        random_state=random_state,
        n_jobs=-1,
        max_depth=12,
    )
    rfe = RFE(estimator=est, n_features_to_select=n_features_to_select, step=step_use)
    rfe.fit(X_imp, labels)
    return X_imp.columns[rfe.support_].tolist()


def permutation_importance_ranking(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_repeats: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance with RandomForest + 72h hit label."""
    labels = hit_within_72_labels(y)
    X_imp = X.fillna(X.median(numeric_only=True)).copy()
    for c in X_imp.select_dtypes(include=["object", "category"]).columns:
        X_imp[c] = pd.factorize(X_imp[c])[0]
    est = RandomForestClassifier(
        n_estimators=150,
        random_state=random_state,
        n_jobs=-1,
        max_depth=12,
    )
    est.fit(X_imp, labels)
    r = permutation_importance(
        est,
        X_imp,
        labels,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_imp.columns,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def compare_top_n_feature_sets(
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_ranking: list[str],
    model_factory: Callable[[], Any],
    n_list: tuple[int, ...] = (20, 30, 50),
    n_splits: int = 5,
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    """Cross-validate ``model_factory`` on top-N features from ``feature_ranking``."""
    from src.models.train import _cross_validate

    out: dict[int, dict[str, float]] = {}
    max_n = len(feature_ranking)
    for n in n_list:
        n_use = min(int(n), max_n)
        cols = feature_ranking[:n_use]
        tag = f"subset_top{n_use}"
        agg = _cross_validate(model_factory, tag, X[cols], y, cols, n_splits, seed)
        out[int(n)] = {
            "brier_mean_cv": agg["brier_mean_mean"],
            "brier_std_cv": agg["brier_mean_std"],
            "n_features": n_use,
        }
    return out


def main() -> None:
    """CLI: RFE + permutation ranking + RF subset CV summary (needs train features)."""
    from pathlib import Path

    from src.models.baselines import get_random_forest_baseline
    from src.utils.config import load_config
    from src.utils.io import read_parquet

    data_config = load_config(Path("configs/data_config.yaml"))
    feat_path = Path(data_config["paths"]["features"]) / data_config["feature_files"]["train"]
    df = read_parquet(feat_path)
    id_col = data_config["id_column"]
    target_col = data_config["target_column"]
    event_col = data_config["event_column"]
    meta = [id_col, target_col, event_col]
    feature_cols = [c for c in df.columns if c not in meta]
    X = df[feature_cols]
    y = df[[target_col, event_col]]

    logger.info("selection_cli", rows=len(X), features=len(feature_cols))

    ranking_pi = permutation_importance_ranking(X, y)
    ranked_list = ranking_pi["feature"].tolist()
    n_rfe = min(40, len(feature_cols))
    rfe_list = recursive_feature_elimination_top(X, y, n_features_to_select=n_rfe)

    out_dir = Path("reports/data_quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_pi.to_csv(out_dir / "permutation_importance.csv", index=False)
    Path(out_dir / "rfe_features.txt").write_text("\n".join(rfe_list), encoding="utf-8")

    subset = compare_top_n_feature_sets(
        X, y, ranked_list, get_random_forest_baseline, n_list=(20, 30, 50)
    )
    lines = ["# Top-N feature subset CV (RandomForest multi-horizon)", ""]
    for n, m in sorted(subset.items()):
        lines.append(
            f"- **N={n}**: brier_mean_cv={m['brier_mean_cv']:.5f} "
            f"± {m['brier_std_cv']:.5f} (features used: {m['n_features']})"
        )
    Path(out_dir / "subset_cv_summary.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info(
        "selection_artifacts",
        permutation_csv=str(out_dir / "permutation_importance.csv"),
        rfe_txt=str(out_dir / "rfe_features.txt"),
        subset_md=str(out_dir / "subset_cv_summary.md"),
    )


if __name__ == "__main__":
    main()
