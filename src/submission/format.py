"""Kaggle submission formatter.

Formats model predictions into the required submission CSV format:
event_id, prob_12h, prob_24h, prob_48h, prob_72h
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import read_parquet

logger = setup_logger(__name__)

HORIZONS = [12, 24, 48, 72]
PROB_COLUMNS = [f"prob_{h}h" for h in HORIZONS]


def validate_submission(df: pd.DataFrame, expected_ids: pd.Series | None = None) -> None:
    """Validate submission DataFrame against Kaggle requirements."""
    config = load_config(Path("configs/data_config.yaml"))
    expected_cols = config["submission_columns"]

    # Check columns
    if list(df.columns) != expected_cols:
        raise ValueError(f"Expected columns {expected_cols}, got {list(df.columns)}")

    # Check no nulls
    if df.isna().any().any():
        raise ValueError("Submission contains null values")

    # Check probabilities in [0, 1]
    for col in PROB_COLUMNS:
        if df[col].min() < 0 or df[col].max() > 1:
            raise ValueError(f"Column {col} has values outside [0, 1]")

    # Check monotonicity: P(12h) <= P(24h) <= P(48h) <= P(72h)
    for i in range(len(HORIZONS) - 1):
        curr = f"prob_{HORIZONS[i]}h"
        nxt = f"prob_{HORIZONS[i + 1]}h"
        violations = (df[nxt] < df[curr] - 1e-9).sum()
        if violations > 0:
            raise ValueError(f"Monotonicity violation: {nxt} < {curr} in {violations} rows")

    # Check row count matches expected IDs
    if expected_ids is not None and len(df) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} rows, got {len(df)}")

    logger.info("submission_valid", rows=len(df))


def _default_submit_model() -> str:
    marker = Path("models/phase5_best_model.txt")
    if marker.is_file():
        name = marker.read_text(encoding="utf-8").strip()
        if name:
            return name
    return "random_forest"


def format_submission(
    model_name: str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Format predictions into Kaggle submission CSV.

    Args:
        model_name: Model whose predictions to format. If None, uses
            ``models/phase5_best_model.txt`` when present, else ``random_forest``.
        output_path: Override output path. If None, auto-generates with date.

    Returns:
        Path to the submission CSV file.
    """
    if model_name is None:
        model_name = _default_submit_model()
    data_config = load_config(Path("configs/data_config.yaml"))
    id_col = data_config["id_column"]

    # Load predictions
    pred_path = Path(data_config["paths"]["predictions"]) / f"predictions_{model_name}.parquet"
    preds = read_parquet(pred_path)
    logger.info("predictions_loaded", path=str(pred_path), rows=len(preds))

    # Build submission
    submission = pd.DataFrame(
        {
            id_col: preds[id_col].astype(np.int64),
        }
    )
    for col in PROB_COLUMNS:
        submission[col] = preds[col].values

    # Enforce monotonicity
    for i in range(1, len(HORIZONS)):
        prev_col = f"prob_{HORIZONS[i - 1]}h"
        curr_col = f"prob_{HORIZONS[i]}h"
        submission[curr_col] = np.maximum(submission[curr_col].values, submission[prev_col].values)

    # Clip to [0, 1]
    for col in PROB_COLUMNS:
        submission[col] = submission[col].clip(0, 1)

    # Validate
    sample_sub = pd.read_csv(
        Path(data_config["paths"]["raw"]) / data_config["raw_files"]["sample_submission"]
    )
    validate_submission(submission, expected_ids=sample_sub[id_col])

    # Save
    if output_path is None:
        sub_dir = Path("submissions")
        sub_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        output_path = sub_dir / f"submission_{model_name}_{today}.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(output_path, index=False)
    logger.info("submission_saved", path=str(output_path), rows=len(submission))
    return output_path


def main() -> None:
    """Run the submission formatting pipeline."""
    logger.info("Formatting submission")
    path = format_submission()
    logger.info("Submission formatted", path=str(path))


if __name__ == "__main__":
    main()
