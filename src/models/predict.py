"""Prediction script.

Loads trained model and test features, generates predictions,
and saves to Parquet.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def predict() -> None:
    """Generate predictions on test set."""
    raise NotImplementedError


def main() -> None:
    """Run the prediction pipeline."""
    logger.info("Starting prediction")
    predict()
    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
