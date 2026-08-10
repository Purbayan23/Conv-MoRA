"""Helpers for capturing environment metadata."""

from __future__ import annotations

import platform
import sys


def environment_summary() -> dict[str, str]:
    """Return a compact environment summary for experiment manifests."""

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
    }
