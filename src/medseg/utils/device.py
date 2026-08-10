"""Device selection helpers."""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency at import time
    torch = None


def detect_device(requested_device: str = "auto") -> str:
    """Resolve the runtime device name."""

    if requested_device != "auto":
        return requested_device

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"
