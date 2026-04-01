"""Training orchestrator.

Loads feature data, applies validation strategy, trains models,
and logs results to MLflow.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def train_model() -> None:
    """Train model with cross-validation and log to MLflow."""
    raise NotImplementedError


def main() -> None:
    """Run the training pipeline."""
    logger.info("Starting model training")
    train_model()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
