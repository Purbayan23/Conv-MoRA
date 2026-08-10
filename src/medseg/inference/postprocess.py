"""Post-processing helpers for segmentation predictions."""

from __future__ import annotations


def resolve_binary_threshold(threshold: float | None, default: float = 0.5) -> float:
    """Resolve the binary threshold used during inference."""

    return default if threshold is None else threshold
