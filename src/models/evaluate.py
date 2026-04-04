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

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- **Mean Brier Score:** {metrics.get('brier_mean', float('nan')):.4f}",
            "",
            "Lower Brier score is better (0 = perfect, 0.25 = random for balanced classes).",
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
