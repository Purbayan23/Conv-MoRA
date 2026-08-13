"""Train the Stage 2 source UNet2D from scratch on ISIC 2016."""

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
from medseg.models import build_model  # noqa: E402
from medseg.optimization import build_optimizer_spec  # noqa: E402
from medseg.training import Trainer  # noqa: E402
from medseg.utils import detect_device, seed_everything  # noqa: E402


def main() -> None:
    """Run the labeled source stage without touching target manifests."""

    overrides = [
        "stage2=convlora_isic2016",
        "model=unet2d_source",
        "loss=bce_reference",
        "optimizer=adam_reference",
        "scheduler=none",
        "experiment=stage2_convlora_isic2016",
        "experiment.name=stage2_source_isic2016",
        *sys.argv[1:],
    ]
    config = compose_config(overrides=overrides)
    seed_everything(config.seed, deterministic=config.runtime.deterministic)
    device = torch.device(detect_device(config.runtime.device))
    config.dataset.loader.batch_size = config.stage2.source_batch_size
    config.optimizer.lr = config.stage2.source_lr
    config.experiment.max_epochs = config.stage2.source_epochs
    loaders = build_dataloaders(config, splits=("train", "val"))
    model = build_model(config).to(device)
    loss_fn = build_loss(config)
    optimizer_spec = build_optimizer_spec(config)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_spec.kwargs)
    run = create_run_directories(config)
    trainer = Trainer.from_config(
        config=config,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
    )
    history = trainer.fit(run_directory=run.root)
    print(f"Source checkpoint: {run.checkpoints / 'best.pt'}")
    print(f"History: {run.root / 'history.json'} ({len(history)} epochs)")


if __name__ == "__main__":
    main()
