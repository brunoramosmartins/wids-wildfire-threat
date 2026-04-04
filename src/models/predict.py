"""Prediction script.

Loads trained model and test features, generates predictions,
and saves to Parquet.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet

logger = setup_logger(__name__)


def predict(model_name: str = "random_forest") -> Path:
    """Generate predictions on test set using a trained MLflow model.

    Args:
        model_name: Name of the MLflow run to load the model from.

    Returns:
        Path to the saved predictions Parquet file.
    """
    config = load_config(Path("configs/model_config.yaml"))
    data_config = load_config(Path("configs/data_config.yaml"))

    # Load test features
    feat_path = Path(data_config["paths"]["features"]) / data_config["feature_files"]["test"]
    df = read_parquet(feat_path)

    id_col = data_config["id_column"]
    feature_cols = [c for c in df.columns if c != id_col]
    X_test = df[feature_cols]
    event_ids = df[id_col]

    logger.info("loading_model", model=model_name)

    # Find the latest run for this model
    experiment_name = config["experiment_name"]
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found. Run training first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(f"No MLflow run found for model '{model_name}'")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    logger.info("model_loaded", run_id=run_id, model=model_name)

    # Generate predictions
    preds = model.predict_proba_horizons(X_test)
    preds.insert(0, id_col, event_ids.values)

    # Save predictions
    pred_path = Path(data_config["paths"]["predictions"]) / f"predictions_{model_name}.parquet"
    write_parquet(preds, pred_path)
    logger.info("predictions_saved", path=str(pred_path), rows=len(preds))

    return pred_path


def main() -> None:
    """Run the prediction pipeline."""
    logger.info("Starting prediction")
    predict()
    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
