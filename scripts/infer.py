"""Evaluate a saved Pure U-Net checkpoint on the separate test split."""

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
from medseg.utils import detect_device  # noqa: E402
from medseg.validation import Evaluator  # noqa: E402
from medseg.inference import Predictor  # noqa: E402


def main() -> None:
    """Generate predictions and evaluate on the separate test split."""

    config = compose_config(overrides=sys.argv[1:])
    if not config.checkpoint.load_path:
        raise SystemExit("Set checkpoint.load_path=/path/to/best.pt to run inference.")

    device = torch.device(detect_device(config.runtime.device))
    try:
        loaders = build_dataloaders(config, splits=("test",))
    except FileNotFoundError as error:
        raise SystemExit(
            "ISIC test data is not available. Extract the official test ZIP files before inference. "
            f"Details: {error}"
        ) from error

    model = build_model(config).to(device)
    predictor = Predictor.from_config(config, model=model, device=device)
    predictor.load(config.checkpoint.load_path)
    prediction_batches = 0
    for batch in loaders["test"]:
        predictor.predict(batch["image"])
        prediction_batches += 1

    results = Evaluator.from_config(config).evaluate(
        model=model,
        dataloader=loaders["test"],
        loss_fn=build_loss(config),
        device=device,
    )
    print(
        f"Test | prediction_batches={prediction_batches} | loss={results['loss']:.4f} | "
        f"dice={results['dice']:.4f} | iou={results['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
