"""Run the BN-statistics-only ISIC 2017 target ablation."""

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
from medseg.data.augmentations import build_image_transform, build_transforms  # noqa: E402
from medseg.data.dataloaders.collate import image_collate, segmentation_collate  # noqa: E402
from medseg.data.datasets import ISIC2017ManifestDataset  # noqa: E402
from medseg.experiment import create_run_directories  # noqa: E402
from medseg.losses import build_loss  # noqa: E402
from medseg.training import (  # noqa: E402
    prepare_bn_only_model,
    run_bn_only,
    split_adaptation_dataset,
)
from medseg.utils import detect_device, seed_everything  # noqa: E402


def main() -> None:
    """Collect BN statistics from target images without labels or gradients."""

    overrides = [
        "stage2=bn_only_isic2017",
        "model=unet2d_source",
        "loss=bce_reference",
        "optimizer=adam_reference",
        "scheduler=none",
        "experiment=bn_only_isic2017",
        *sys.argv[1:],
    ]
    config = compose_config(overrides=overrides)
    if config.stage2.adaptation_mode != "bn_only":
        raise SystemExit("This entry point requires stage2.adaptation_mode=bn_only.")
    if not config.stage2.source_checkpoint:
        raise SystemExit("Set stage2.source_checkpoint=/path/to/source/best.pt.")

    seed_everything(config.seed, deterministic=config.runtime.deterministic)
    device = torch.device(detect_device(config.runtime.device))

    adaptation_dataset = ISIC2017ManifestDataset(
        config=config,
        manifest_path=config.stage2.target_adaptation_manifest,
        images_dir=config.stage2.target_images_dir,
        masks_dir=None,
        transform=build_image_transform(config, stage="eval"),
        include_mask=False,
    )
    if len(adaptation_dataset) != 1003:
        raise RuntimeError("Expected exactly 1003 images in the target adaptation manifest.")
    adaptation_subset, consistency_subset, adaptation_indices, consistency_indices = (
        split_adaptation_dataset(
            adaptation_dataset,
            consistency_fraction=config.stage2.consistency_fraction,
            seed=config.stage2.consistency_split_seed,
        )
    )
    if len(adaptation_subset) != 803 or len(consistency_subset) != 200:
        raise RuntimeError(
            "Unexpected internal target split; expected 803 adaptation and 200 consistency samples."
        )
    if set(adaptation_indices).intersection(consistency_indices):
        raise RuntimeError("Adaptation and consistency subsets must be disjoint.")
    if set(adaptation_indices).union(consistency_indices) != set(range(len(adaptation_dataset))):
        raise RuntimeError("The internal target split does not cover all adaptation-manifest rows.")

    evaluation_dataset = ISIC2017ManifestDataset(
        config=config,
        manifest_path=config.stage2.target_eval_manifest,
        images_dir=config.stage2.target_images_dir,
        masks_dir=config.stage2.target_masks_dir,
        transform=build_transforms(config, stage="eval"),
        include_mask=True,
    )
    if len(evaluation_dataset) != 251:
        raise RuntimeError("Expected exactly 251 images in the target evaluation manifest.")
    adaptation_loader = DataLoader(
        adaptation_subset,
        batch_size=config.stage2.adaptation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.runtime.num_workers or 0,
        pin_memory=bool(config.runtime.pin_memory),
        collate_fn=image_collate,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.stage2.adaptation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.runtime.num_workers or 0,
        pin_memory=bool(config.runtime.pin_memory),
        collate_fn=segmentation_collate,
    )

    model = prepare_bn_only_model(
        config,
        source_checkpoint=config.stage2.source_checkpoint,
        device=device,
    )
    run = create_run_directories(config)
    final_path = run_bn_only(
        config=config,
        model=model,
        adaptation_dataloader=adaptation_loader,
        evaluation_dataloader=evaluation_loader,
        loss_fn=build_loss(config),
        device=device,
        output_dir=run.root,
        consistency_subset_size=len(consistency_subset),
    )
    print(f"BN-only final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
