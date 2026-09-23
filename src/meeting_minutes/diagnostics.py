"""Safe server-side diagnostics for external provider failures."""

from __future__ import annotations

import logging
import os
import re
from typing import Any


_SECRET_PATTERN = re.compile(
    r"(?i)(?:bearer\s+)?sk-[a-z0-9_-]{8,}|"
    r"(?:api[_-]?key|token|secret)\s*[:=]\s*[^\s,;]+"
)


def log_provider_error(
    logger: logging.Logger,
    *,
    operation: str,
    model: str,
    error: Exception,
) -> None:
    """Log actionable provider metadata while redacting credentials."""
    body = getattr(error, "body", None)
    body_error = body.get("error", body) if isinstance(body, dict) else {}
    code = getattr(error, "code", None)
    if not code and isinstance(body_error, dict):
        code = body_error.get("code")

    cause = getattr(error, "__cause__", None)
    cause_type = type(cause).__name__ if cause is not None else None
    cause_message = _sanitize_message(cause) if cause is not None else None

    logger.error(
        "%s failed: model=%s error_type=%s status_code=%s code=%s "
        "request_id=%s message=%s cause_type=%s cause=%s",
        operation,
        model,
        type(error).__name__,
        getattr(error, "status_code", None),
        code,
        getattr(error, "request_id", None),
        _sanitize_message(error),
        cause_type,
        cause_message,
    )


def _sanitize_message(value: Any) -> str:
    """Return a one-line, bounded message with known secret values removed."""
    message = " ".join(str(value).split())
    for name, secret in os.environ.items():
        upper_name = name.upper()
        if secret and any(marker in upper_name for marker in ("KEY", "TOKEN", "SECRET")):
            message = message.replace(secret, "[REDACTED]")
    message = _SECRET_PATTERN.sub("[REDACTED]", message)
    return message[:1_000]
