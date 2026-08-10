"""Dataset builder entry points."""

from __future__ import annotations

from medseg.config.schema import AppConfig, DatasetConfig
from medseg.data.datasets.base import BaseSegmentationDataset, SampleTransform
from medseg.data.datasets.binary_folder import BinaryFolderSegmentationDataset
from medseg.data.datasets.isic2016 import ISIC2016Dataset


def build_dataset(
    config: AppConfig | DatasetConfig,
    split: str,
    transform: SampleTransform | None = None,
) -> BaseSegmentationDataset:
    """Build the configured dataset adapter for one split."""

    dataset_config = config.dataset if isinstance(config, AppConfig) else config
    if dataset_config.adapter == "isic2016":
        return ISIC2016Dataset(config=dataset_config, split=split, transform=transform)
    if dataset_config.adapter == "binary_folder":
        return BinaryFolderSegmentationDataset(config=dataset_config, split=split, transform=transform)
    raise ValueError(f"Unknown dataset adapter '{dataset_config.adapter}'.")
