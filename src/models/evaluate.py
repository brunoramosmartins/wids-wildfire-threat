"""Evaluation and MLflow logging.

Implements the WiDS 2026 official metric:

    Hybrid = 0.3 * C_index + 0.7 * (1 - Weighted_Brier)

Weighted_Brier = 0.3 * Brier@24h + 0.4 * Brier@48h + 0.3 * Brier@72h

Per-horizon Brier uses **censor-aware** labels (censored-before-H is
EXCLUDED, not counted as 0) — see docs/problem_statement.md.

All logging and reports surface the Hybrid Score as the primary metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.observability.logger import setup_logger
from src.utils.config import load_config

logger = setup_logger(__name__)

HORIZONS = [12, 24, 48, 72]

# Weighted Brier — 48h highest per competition description
# (12h is NOT in the scored metric, only in the submission format)
SCORED_HORIZONS = [24, 48, 72]
BRIER_WEIGHTS = {24: 0.3, 48: 0.4, 72: 0.3}
# Sanity check at import time
assert abs(sum(BRIER_WEIGHTS.values()) - 1.0) < 1e-9


def censor_aware_brier_at_horizon(
    y_true: pd.DataFrame,
    prob: np.ndarray,
    horizon: float,
) -> float:
    """Censor-aware Brier score at a single horizon.

    Inclusion rules per the competition description:

    - Hit within H (event=1, T<=H) -> label=1, included
    - Hit after H  (event=1, T>H)  -> label=0, included
    - Censored after H  (event=0, T>H)  -> label=0, included
    - Censored before H (event=0, T<H)  -> EXCLUDED

    Returns NaN if no samples remain (should not happen in practice).
    """
    T = np.asarray(y_true["time_to_hit_hours"].values, dtype=float)
    E = np.asarray(y_true["event"].values, dtype=int)
    pred = np.asarray(prob, dtype=float)

    # Exclude censored-before-horizon
    include_mask = ~((E == 0) & (T < horizon))
    if not include_mask.any():
        return float("nan")

    label = ((E == 1) & (T <= horizon)).astype(np.int32)
    label_in = label[include_mask]
    pred_in = pred[include_mask]

    # brier_score_loss requires both classes, but it works fine with one; safeguard anyway
    return float(np.mean((pred_in - label_in) ** 2))


def weighted_brier_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> tuple[float, dict[int, float]]:
    """Competition-weighted Brier: 0.3 B@24 + 0.4 B@48 + 0.3 B@72.

    Returns (weighted_brier, per_horizon_dict).
    """
    per_horizon: dict[int, float] = {}
    weighted = 0.0
    for h in SCORED_HORIZONS:
        b = censor_aware_brier_at_horizon(y_true, y_pred[f"prob_{h}h"].values, float(h))
        per_horizon[h] = b
        weighted += BRIER_WEIGHTS[h] * b
    return float(weighted), per_horizon


def harrell_c_index(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    risk_col: str = "prob_72h",
) -> float:
    """Harrell's concordance index using `prob_72h` as the scalar risk.

    Higher predicted `prob_72h` = higher risk = sooner event expected.
    """
    from sksurv.metrics import concordance_index_censored

    ev = np.asarray(y_true["event"].values, dtype=bool)
    tt = np.asarray(y_true["time_to_hit_hours"].values, dtype=float)
    risk = np.asarray(y_pred[risk_col].values, dtype=float)
    c, *_ = concordance_index_censored(ev, tt, risk)
    return float(c)


def hybrid_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """Official WiDS 2026 Hybrid metric.

    Returns (hybrid, components) where components includes all pieces
    the metric depends on.
    """
    wb, per_h = weighted_brier_score(y_true, y_pred)
    ci = harrell_c_index(y_true, y_pred)
    hybrid = 0.3 * ci + 0.7 * (1.0 - wb)
    components = {
        "c_index": ci,
        "weighted_brier": wb,
        "brier_24h_ca": per_h[24],
        "brier_48h_ca": per_h[48],
        "brier_72h_ca": per_h[72],
        "hybrid_score": hybrid,
    }
    return float(hybrid), components


def compute_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> dict[str, float]:
    """Compute official metric + diagnostics.

    Primary: Hybrid Score (higher = better). Also emits censor-aware Brier
    per horizon, C-index, plus legacy naive-Brier/AUC/logloss for
    backward-compat diagnostics.
    """
    metrics: dict[str, float] = {}
    hybrid, comps = hybrid_score(y_true, y_pred)
    metrics.update({k: float(v) for k, v in comps.items()})

    # Legacy naive-Brier + AUC + logloss per horizon (diagnostic only)
    T = np.asarray(y_true["time_to_hit_hours"].values, dtype=float)
    E = np.asarray(y_true["event"].values, dtype=int)
    for h in HORIZONS:
        label = ((E == 1) & (T <= h)).astype(int)
        pred = y_pred[f"prob_{h}h"].values
        metrics[f"brier_{h}h_naive"] = float(brier_score_loss(label, pred))
        if len(np.unique(label)) > 1:
            metrics[f"auc_{h}h"] = float(roc_auc_score(label, pred))
        try:
            metrics[f"logloss_{h}h"] = float(log_loss(label, pred))
        except ValueError:
            pass

    # Calibration gap at 72h (diagnostic only)
    try:
        label72 = ((E == 1) & (T <= 72)).astype(int)
        pred72 = y_pred["prob_72h"].values.astype(float)
        if len(np.unique(label72)) > 1 and len(label72) >= 40:
            frac_pos, mean_pred = calibration_curve(label72, pred72, n_bins=8, strategy="quantile")
            metrics["calibration_gap_72h"] = float(np.mean(np.abs(frac_pos - mean_pred)))
    except Exception:
        pass

    # Keep brier_mean (naive, 4 horizons) for backward-compat with Phase 4
    # dashboards / older tests.
    metrics["brier_mean"] = float(np.mean([metrics[f"brier_{h}h_naive"] for h in HORIZONS]))

    return metrics


def log_to_mlflow(
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    feature_list: list[str],
    model_artifact: Any | None = None,
) -> str:
    """Log experiment run to MLflow.

    Returns the run_id.
    """
    config = load_config(Path("configs/model_config.yaml"))
    experiment_name = config["experiment_name"]
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_text("\n".join(feature_list), "feature_list.txt")

        if model_artifact is not None:
            mlflow.sklearn.log_model(model_artifact, "model")

        run_id: str = run.info.run_id
        logger.info(
            "mlflow_logged",
            model=model_name,
            run_id=run_id,
            hybrid=round(metrics.get("hybrid_score", float("nan")), 5),
            weighted_brier=round(metrics.get("weighted_brier", float("nan")), 5),
            c_index=round(metrics.get("c_index", float("nan")), 5),
        )
        return run_id


def generate_report(
    model_name: str,
    metrics: dict[str, float],
    output_dir: Path | None = None,
) -> Path:
    """Generate a markdown evaluation report anchored on Hybrid Score."""
    if output_dir is None:
        output_dir = Path("reports/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Evaluation Report: {model_name}",
        "",
        "## WiDS 2026 Official Metric",
        "",
        f"- **Hybrid Score:** {metrics.get('hybrid_score', float('nan')):.5f}  *(higher = better)*",
        f"- **C-index (30%):** {metrics.get('c_index', float('nan')):.5f}",
        f"- **Weighted Brier (70%):** {metrics.get('weighted_brier', float('nan')):.5f}  "
        f"*(lower = better)*",
        "",
        "## Censor-aware Brier per scored horizon",
        "",
        "| Horizon | Weight | Censor-aware Brier |",
        "|---------|--------|--------------------|",
        f"| 24h | 0.3 | {metrics.get('brier_24h_ca', float('nan')):.5f} |",
        f"| 48h | 0.4 | {metrics.get('brier_48h_ca', float('nan')):.5f} |",
        f"| 72h | 0.3 | {metrics.get('brier_72h_ca', float('nan')):.5f} |",
        "",
        "## Diagnostics (not scored)",
        "",
        "| Horizon | Naive Brier | AUC | Log Loss |",
        "|---------|-------------|-----|----------|",
    ]

    for h in HORIZONS:
        brier = metrics.get(f"brier_{h}h_naive", float("nan"))
        auc = metrics.get(f"auc_{h}h", float("nan"))
        ll = metrics.get(f"logloss_{h}h", float("nan"))
        lines.append(f"| {h}h | {brier:.5f} | {auc:.5f} | {ll:.5f} |")

    if "calibration_gap_72h" in metrics:
        lines.extend(
            [
                "",
                f"- **Calibration gap @72h (decile-weighted):** "
                f"{metrics['calibration_gap_72h']:.4f} (lower is better)",
            ]
        )

    path = output_dir / f"evaluation_{model_name}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report_generated", path=str(path), model=model_name)
    return path


def evaluate_model(
    model_name: str,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    params: dict[str, Any],
    feature_list: list[str],
    model_artifact: Any | None = None,
) -> dict[str, float]:
    """Full evaluation: compute metrics, log to MLflow, generate report."""
    metrics = compute_metrics(y_true, y_pred)
    log_to_mlflow(model_name, params, metrics, feature_list, model_artifact)
    generate_report(model_name, metrics)
    return metrics


def main() -> None:
    """Run the evaluation pipeline (standalone)."""
    logger.info("Evaluation module ready — call evaluate_model() from train.py")


if __name__ == "__main__":
    main()
