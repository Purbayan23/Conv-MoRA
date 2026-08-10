"""Visualization naming helpers for qualitative results."""

from __future__ import annotations


def default_overlay_name(sample_id: str) -> str:
    """Return a deterministic filename for one qualitative overlay."""

    return f"{sample_id}_overlay.png"
