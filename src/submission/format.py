"""Kaggle submission formatter.

Formats model predictions into the required submission CSV format.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def format_submission() -> None:
    """Format predictions into Kaggle submission CSV."""
    raise NotImplementedError


def main() -> None:
    """Run the submission formatting pipeline."""
    logger.info("Formatting submission")
    format_submission()
    logger.info("Submission formatted")


if __name__ == "__main__":
    main()
