"""Pipeline health checks.

Verifies expected files exist, row counts are consistent,
and predictions are within expected ranges.
"""

from __future__ import annotations

from pathlib import Path


def check_file_exists(path: Path) -> bool:
    """Check that an expected output file exists."""
    return path.exists()


def check_row_count(actual: int, expected: int, tolerance: float = 0.01) -> bool:
    """Check that row count is within tolerance of expected."""
    return abs(actual - expected) <= expected * tolerance
