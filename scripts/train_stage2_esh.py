"""Train the frozen-source Early Segmentation Head for Stage 2."""

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
from medseg.experiment import create_run_directories  # noqa: E402
from medseg.losses import build_loss  # noqa: E402
from medseg.training import train_esh  # noqa: E402
from medseg.utils import detect_device, seed_everything  # noqa: E402


def main() -> None:
    """Train ESH using only labeled source train/validation splits."""

    overrides = [
        "stage2=convlora_isic2016",
        "model=unet2d_source",
        "loss=bce_reference",
        "optimizer=adam_reference",
        "scheduler=none",
        "experiment=stage2_convlora_isic2016",
        "experiment.name=stage2_esh_isic2016",
        *sys.argv[1:],
    ]
    config = compose_config(overrides=overrides)
    if not config.stage2.source_checkpoint:
        raise SystemExit("Set stage2.source_checkpoint=/path/to/source/best.pt.")
    seed_everything(config.seed, deterministic=config.runtime.deterministic)
    device = torch.device(detect_device(config.runtime.device))
    config.dataset.loader.batch_size = config.stage2.esh_batch_size
    loaders = build_dataloaders(config, splits=("train", "val"))
    run = create_run_directories(config)
    best_path = train_esh(
        config=config,
        source_checkpoint=config.stage2.source_checkpoint,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        loss_fn=build_loss(config),
        device=device,
        output_dir=run.checkpoints,
    )
    print(f"ESH checkpoint: {best_path}")


if __name__ == "__main__":
    main()
