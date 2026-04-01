"""Structured logging setup.

Configures structlog with JSON output and context binding.
"""

from __future__ import annotations

import logging

import structlog


def setup_logger(name: str) -> structlog.BoundLogger:
    """Configure and return a structured logger."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logger: structlog.BoundLogger = structlog.get_logger(name)
    return logger
