"""Feature engineering orchestrator.

Loads processed data, applies all feature modules, combines into
a single feature matrix, and saves to Parquet.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_features() -> None:
    """Build feature matrices for train and test datasets."""
    raise NotImplementedError


def main() -> None:
    """Run the feature engineering pipeline."""
    logger.info("Starting feature engineering")
    build_features()
    logger.info("Feature engineering complete")


if __name__ == "__main__":
    main()
