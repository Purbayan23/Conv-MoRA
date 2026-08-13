"""Adapt encoder ConvLoRA parameters on the unlabeled ISIC 2017 target split."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import compose_config  # noqa: E402
from medseg.data.augmentations import build_image_transform  # noqa: E402
from medseg.data.dataloaders.collate import image_collate  # noqa: E402
from medseg.data.datasets import ISIC2017ManifestDataset  # noqa: E402
from medseg.experiment import create_run_directories  # noqa: E402
from medseg.losses import build_loss  # noqa: E402
from medseg.training import (  # noqa: E402
    adapt_model,
    prepare_adaptation_model,
    prepare_frozen_esh,
    split_adaptation_dataset,
)
from medseg.utils import detect_device, seed_everything  # noqa: E402


def main() -> None:
    """Run target adaptation without opening target masks."""

    overrides = [
        "stage2=convlora_isic2016",
        "model=unet2d_convlora",
        "loss=bce_reference",
        "optimizer=adam_reference",
        "scheduler=none",
        "experiment=stage2_convlora_isic2016",
        "experiment.name=stage2_adaptation_isic2017",
        *sys.argv[1:],
    ]
    config = compose_config(overrides=overrides)
    if not config.stage2.source_checkpoint:
        raise SystemExit("Set stage2.source_checkpoint=/path/to/source/best.pt.")
    if not config.stage2.esh_checkpoint:
        raise SystemExit("Set stage2.esh_checkpoint=/path/to/esh/best.pt.")
    seed_everything(config.seed, deterministic=config.runtime.deterministic)
    device = torch.device(detect_device(config.runtime.device))
    dataset = ISIC2017ManifestDataset(
        config=config,
        manifest_path=config.stage2.target_adaptation_manifest,
        images_dir=config.stage2.target_images_dir,
        masks_dir=config.stage2.target_masks_dir,
        transform=build_image_transform(config, stage="eval"),
        include_mask=False,
    )
    adaptation_dataset, consistency_dataset, adaptation_indices, consistency_indices = (
        split_adaptation_dataset(
            dataset,
            consistency_fraction=config.stage2.consistency_fraction,
            seed=config.stage2.consistency_split_seed,
        )
    )
    loader = DataLoader(
        adaptation_dataset,
        batch_size=config.stage2.adaptation_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.runtime.num_workers or 0,
        pin_memory=bool(config.runtime.pin_memory),
        collate_fn=image_collate,
    )
    consistency_loader = DataLoader(
        consistency_dataset,
        batch_size=config.stage2.adaptation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.runtime.num_workers or 0,
        pin_memory=bool(config.runtime.pin_memory),
        collate_fn=image_collate,
    )
    if set(adaptation_indices).intersection(consistency_indices):
        raise RuntimeError("Adaptation and consistency subsets must be disjoint.")
    model = prepare_adaptation_model(
        config,
        source_checkpoint=config.stage2.source_checkpoint,
        device=device,
    )
    esh = prepare_frozen_esh(config, config.stage2.esh_checkpoint, device=device)
    run = create_run_directories(config)
    best_path = adapt_model(
        config=config,
        model=model,
        esh=esh,
        dataloader=loader,
        loss_fn=build_loss(config),
        device=device,
        output_dir=run.checkpoints,
        consistency_dataloader=consistency_loader,
    )
    print(f"Adaptation checkpoint: {best_path}")


if __name__ == "__main__":
    main()
