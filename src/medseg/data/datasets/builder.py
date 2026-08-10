"""Dataset builder entry points."""

from __future__ import annotations

from typing import Any

from medseg.data.augmentations.builder import build_transforms
from medseg.data.datasets.base import BaseSegmentationDataset, SampleTransform
from medseg.data.datasets.binary_folder import BinaryFolderSegmentationDataset
from medseg.data.datasets.isic2016 import ISIC2016Dataset


def build_dataset(
    config: Any,
    split: str,
    transform: SampleTransform | None = None,
) -> BaseSegmentationDataset:
    """Build the configured dataset adapter for one split."""

    dataset_config = getattr(config, "dataset", None)
    if dataset_config is not None:
        if transform is None:
            transform = build_transforms(config, stage="train" if split == "train" else "eval")
    else:
        dataset_config = config
        if transform is None:
            raise ValueError(
                "A transform pipeline is required when building a dataset from DatasetConfig only."
            )

    if dataset_config.adapter == "isic2016":
        return ISIC2016Dataset(config=dataset_config, split=split, transform=transform)
    if dataset_config.adapter == "binary_folder":
        return BinaryFolderSegmentationDataset(config=dataset_config, split=split, transform=transform)
    raise ValueError(f"Unknown dataset adapter '{dataset_config.adapter}'.")
