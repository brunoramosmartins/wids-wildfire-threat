"""Evaluation and MLflow logging.

Computes competition and secondary metrics, logs to MLflow,
and generates evaluation reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.observability.logger import setup_logger
from src.utils.config import load_config

logger = setup_logger(__name__)

HORIZONS = [12, 24, 48, 72]


def compute_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> dict[str, float]:
    """Compute per-horizon and aggregate metrics.

    Args:
        y_true: DataFrame with time_to_hit_hours and event columns.
        y_pred: DataFrame with prob_12h, prob_24h, prob_48h, prob_72h columns.

    Returns:
        Dictionary of metric_name -> value.
    """
    T = y_true["time_to_hit_hours"].values
    E = y_true["event"].values
    metrics: dict[str, float] = {}

    brier_scores = []
    for h in HORIZONS:
        # Binary label: hit within h hours
        label = ((E == 1) & (T <= h)).astype(int)
        pred = y_pred[f"prob_{h}h"].values

        brier = float(brier_score_loss(label, pred))
        metrics[f"brier_{h}h"] = brier
        brier_scores.append(brier)

        # AUC only if both classes present
        if len(np.unique(label)) > 1:
            auc = float(roc_auc_score(label, pred))
            metrics[f"auc_{h}h"] = auc

        # Log loss
        try:
            ll = float(log_loss(label, pred))
            metrics[f"logloss_{h}h"] = ll
        except ValueError:
            pass

    metrics["brier_mean"] = float(np.mean(brier_scores))

    # Concordance index (IPCW-free): higher predicted P(hit by 72h) ⇒ higher event risk
    try:
        from sksurv.metrics import concordance_index_censored

        ev = y_true["event"].values.astype(bool)
        tt = y_true["time_to_hit_hours"].values.astype(float)
        risk = y_pred["prob_72h"].values.astype(float)
        ci, *_ = concordance_index_censored(ev, tt, risk)
        metrics["c_index"] = float(ci)
    except Exception:
        pass

    # Calibration gap at 72h (mean |fraction positive − mean pred| per quantile bin)
    try:
        from sklearn.calibration import calibration_curve

        label72 = ((E == 1) & (T <= 72)).astype(int)
        pred72 = y_pred["prob_72h"].values.astype(float)
        if len(np.unique(label72)) > 1 and len(label72) >= 40:
            frac_pos, mean_pred = calibration_curve(label72, pred72, n_bins=8, strategy="quantile")
            metrics["calibration_gap_72h"] = float(np.mean(np.abs(frac_pos - mean_pred)))
    except Exception:
        pass

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
            metrics={k: round(v, 4) for k, v in metrics.items()},
        )
        return run_id


def generate_report(
    model_name: str,
    metrics: dict[str, float],
    output_dir: Path | None = None,
) -> Path:
    """Generate a markdown evaluation report."""
    if output_dir is None:
        output_dir = Path("reports/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Evaluation Report: {model_name}",
        "",
        "## Per-Horizon Metrics",
        "",
        "| Horizon | Brier Score | AUC | Log Loss |",
        "|---------|-------------|-----|----------|",
    ]

    for h in HORIZONS:
        brier = metrics.get(f"brier_{h}h", float("nan"))
        auc = metrics.get(f"auc_{h}h", float("nan"))
        ll = metrics.get(f"logloss_{h}h", float("nan"))
        lines.append(f"| {h}h | {brier:.4f} | {auc:.4f} | {ll:.4f} |")

    agg_lines = [
        "",
        "## Aggregate",
        "",
        f"- **Mean Brier Score:** {metrics.get('brier_mean', float('nan')):.4f}",
        "",
        "Lower Brier score is better (0 = perfect, 0.25 = random for balanced classes).",
    ]
    if "c_index" in metrics:
        agg_lines.extend(
            [
                "",
                f"- **Harrell C-index (risk = prob_72h):** {metrics['c_index']:.4f}",
                "  (1.0 = perfect discrimination on censored outcomes; 0.5 ≈ random.)",
            ]
        )
    if "calibration_gap_72h" in metrics:
        agg_lines.append(
            f"- **Calibration gap (72h, decile-weighted):** "
            f"{metrics['calibration_gap_72h']:.4f} (lower is better)"
        )
    lines.extend(agg_lines)

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
