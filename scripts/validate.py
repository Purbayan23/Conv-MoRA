"""Evaluate a saved Pure U-Net checkpoint on the validation split."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import compose_config  # noqa: E402
from medseg.data.dataloaders.builder import build_dataloaders  # noqa: E402
from medseg.losses import build_loss  # noqa: E402
from medseg.models import build_model  # noqa: E402
from medseg.training.checkpointing import load_checkpoint  # noqa: E402
from medseg.utils import detect_device  # noqa: E402
from medseg.validation import Evaluator  # noqa: E402


def main() -> None:
    """Evaluate the configured checkpoint on validation data."""

    config = compose_config(overrides=sys.argv[1:])
    if not config.checkpoint.load_path:
        raise SystemExit("Set checkpoint.load_path=/path/to/best.pt to run validation.")

    device = torch.device(detect_device(config.runtime.device))
    loaders = build_dataloaders(config, splits=("val",))
    model = build_model(config).to(device)
    load_checkpoint(Path(config.checkpoint.load_path), model, device=device)
    results = Evaluator.from_config(config).evaluate(
        model=model,
        dataloader=loaders["val"],
        loss_fn=build_loss(config),
        device=device,
    )
    print(
        f"Validation | loss={results['loss']:.4f} | "
        f"dice={results['dice']:.4f} | iou={results['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
