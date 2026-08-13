"""Evaluate a Stage 2 adapted model on the frozen ISIC 2017 target split."""

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
from medseg.data.augmentations import build_transforms  # noqa: E402
from medseg.data.dataloaders.collate import segmentation_collate  # noqa: E402
from medseg.data.datasets import ISIC2017ManifestDataset  # noqa: E402
from medseg.losses import build_loss  # noqa: E402
from medseg.training import load_model_state, prepare_adaptation_model  # noqa: E402
from medseg.utils import detect_device  # noqa: E402
from medseg.validation import Evaluator  # noqa: E402


def main() -> None:
    """Evaluate with target masks that are never used during adaptation."""

    overrides = [
        "stage2=convlora_isic2016",
        "model=unet2d_convlora",
        "loss=bce_reference",
        "optimizer=adam_reference",
        "scheduler=none",
        "experiment=stage2_convlora_isic2016",
        *sys.argv[1:],
    ]
    config = compose_config(overrides=overrides)
    if not config.stage2.adaptation_checkpoint:
        raise SystemExit("Set stage2.adaptation_checkpoint=/path/to/adaptation/best.pt.")
    device = torch.device(detect_device(config.runtime.device))
    dataset = ISIC2017ManifestDataset(
        config=config,
        manifest_path=config.stage2.target_eval_manifest,
        images_dir=config.stage2.target_images_dir,
        masks_dir=config.stage2.target_masks_dir,
        transform=build_transforms(config, stage="eval"),
        include_mask=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.stage2.adaptation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.runtime.num_workers or 0,
        pin_memory=bool(config.runtime.pin_memory),
        collate_fn=segmentation_collate,
    )
    model = prepare_adaptation_model(
        config,
        source_checkpoint=config.stage2.source_checkpoint,
        device=device,
    )
    load_model_state(config.stage2.adaptation_checkpoint, model, device=device)
    model.eval()
    results = Evaluator.from_config(config).evaluate(
        model=model,
        dataloader=loader,
        loss_fn=build_loss(config),
        device=device,
    )
    print(
        f"ISIC2017 target eval | samples={len(dataset)} | "
        f"loss={results['loss']:.4f} | dice={results['dice']:.4f} | "
        f"iou={results['iou']:.4f} | precision={results['precision']:.4f} | "
        f"recall={results['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
