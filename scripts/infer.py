"""Inference entry point scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import compose_config  # noqa: E402


def main() -> None:
    """Compose and print the inference config."""

    config = compose_config(overrides=sys.argv[1:])
    print(OmegaConf.to_yaml(config, resolve=True))
    print("Inference scaffold created. Predictor implementation is intentionally pending.")


if __name__ == "__main__":
    main()
