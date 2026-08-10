"""Visualization naming helpers for scalar training curves."""

from __future__ import annotations


def default_curve_name(metric_name: str) -> str:
    """Return a deterministic filename for one scalar curve plot."""

    return f"{metric_name}_curve.png"
