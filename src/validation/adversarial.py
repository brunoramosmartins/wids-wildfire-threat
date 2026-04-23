"""Adversarial validation train-vs-test (Phase 6.5 diagnostic).

Fits a classifier that tries to distinguish training rows from test rows.
Reports 5-fold AUC on the combined set and top-important features. If AUC
> ~0.60, the train and test distributions differ meaningfully and some
form of covariate-shift correction (sample reweighting, feature removal,
or stratified subsetting) is likely worth exploring.

Output: ``reports/data_quality/adversarial_validation.md`` and
``reports/data_quality/adversarial_feature_importance.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import read_parquet

logger = setup_logger(__name__)


def run_adversarial_validation() -> dict:
    config = load_config(Path("configs/data_config.yaml"))
    feat_dir = Path(config["paths"]["features"])
    train = read_parquet(feat_dir / config["feature_files"]["train"])
    test = read_parquet(feat_dir / config["feature_files"]["test"])

    id_col = config["id_column"]
    target_col = config["target_column"]
    event_col = config["event_column"]
    meta = {id_col, target_col, event_col}
    feature_cols = [c for c in test.columns if c not in meta]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    # Fill any NaN with column median from combined
    combined_medians = pd.concat([X_train, X_test], axis=0).median(numeric_only=True)
    X_train = X_train.fillna(combined_medians)
    X_test = X_test.fillna(combined_medians)

    # Assemble adversarial dataset: label = 1 if from test, 0 if from train
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)])

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
    )
    auc_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    mean_auc = float(np.mean(auc_scores))
    std_auc = float(np.std(auc_scores))

    # Feature importance from a single fit
    clf.fit(X, y)
    imp = pd.DataFrame(
        {"feature": feature_cols, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)

    out_dir = Path("reports/data_quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_dir / "adversarial_feature_importance.csv", index=False)

    # Interpretive verdict
    if mean_auc < 0.55:
        verdict = "No meaningful shift — train/test look drawn from the same distribution."
    elif mean_auc < 0.65:
        verdict = (
            "Mild shift — some feature distributions differ. Consider checking the top "
            "features below and evaluating `CovariateShiftWarning` on those columns."
        )
    else:
        verdict = (
            "Significant shift — model can separate train from test with high confidence. "
            "Strongly consider sample reweighting (propensity weighting) or removing the "
            "highest-importance discriminating features."
        )

    lines = [
        "# Adversarial Validation — Train vs. Test",
        "",
        "Cross-validated AUC of a RandomForest distinguishing train from test rows.",
        "",
        f"- **CV AUC:** {mean_auc:.4f} ± {std_auc:.4f}",
        f"- **Interpretation:** {verdict}",
        "",
        "## Top 15 most discriminating features",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for i, (_, row) in enumerate(imp.head(15).iterrows(), start=1):
        lines.append(f"| {i} | `{row['feature']}` | {row['importance']:.4f} |")
    lines.append("")
    lines.append(f"Full ranking in `{out_dir / 'adversarial_feature_importance.csv'}`.")

    md_path = out_dir / "adversarial_validation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "auc_mean": mean_auc,
        "auc_std": std_auc,
        "verdict": verdict,
        "top_features": imp.head(15).to_dict(orient="records"),
    }
    (out_dir / "adversarial_validation.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "adversarial_validation_done",
        auc_mean=round(mean_auc, 4),
        auc_std=round(std_auc, 4),
        verdict=verdict[:80],
    )
    return summary


def main() -> None:
    run_adversarial_validation()


if __name__ == "__main__":
    main()
