"""Feature engineering orchestrator.

Loads processed data, applies all feature modules, drops redundant
columns, combines into a single feature matrix, and saves to Parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.geospatial import compute_geospatial_features
from src.features.infrastructure import compute_infrastructure_features
from src.features.temporal import compute_temporal_features
from src.features.weather import compute_weather_features
from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet

logger = setup_logger(__name__)

REDUNDANT_COLUMNS = [
    "relative_growth_0_5h",
    "closing_speed_abs_m_per_h",
    "radial_growth_rate_m_per_h",
]


def _engineer_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Apply all feature modules to a single dataset."""
    logger.info("engineering_features", dataset=dataset_name, input_cols=len(df.columns))

    # Compute domain features
    geo = compute_geospatial_features(df)
    temp = compute_temporal_features(df)
    weather = compute_weather_features(df)
    infra = compute_infrastructure_features(df)

    logger.info(
        "feature_groups",
        geospatial=len(geo.columns),
        temporal=len(temp.columns),
        weather=len(weather.columns),
        infrastructure=len(infra.columns),
        dataset=dataset_name,
    )

    # Combine: base + engineered
    result = pd.concat([df, geo, temp, weather, infra], axis=1)

    # Drop redundant columns
    to_drop = [c for c in REDUNDANT_COLUMNS if c in result.columns]
    if to_drop:
        result = result.drop(columns=to_drop)
        logger.info("dropped_redundant", columns=to_drop, dataset=dataset_name)

    logger.info(
        "features_built",
        output_cols=len(result.columns),
        rows=len(result),
        dataset=dataset_name,
    )
    return result


def build_features() -> None:
    """Build feature matrices for train and test datasets."""
    config = load_config(Path("configs/data_config.yaml"))

    proc_dir = Path(config["paths"]["processed"])
    feat_dir = Path(config["paths"]["features"])

    for split in ("train", "test"):
        proc_path = proc_dir / config["processed_files"][split]
        feat_path = feat_dir / config["feature_files"][split]

        df = read_parquet(proc_path)
        features = _engineer_features(df, split)
        write_parquet(features, feat_path)
        logger.info("saved_features", path=str(feat_path), dataset=split)


def main() -> None:
    """Run the feature engineering pipeline."""
    logger.info("feature_pipeline_start")
    build_features()
    logger.info("feature_pipeline_complete")


if __name__ == "__main__":
    main()
