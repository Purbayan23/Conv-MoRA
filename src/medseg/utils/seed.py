"""Reproducibility helpers for deterministic experiments."""

from __future__ import annotations

import os
import random

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency at import time
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency at import time
    torch = None


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed supported random number generators."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
