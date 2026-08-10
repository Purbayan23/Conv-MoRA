"""ISIC 2016 dataset scaffold built on the generic binary dataset contract."""

from __future__ import annotations

from medseg.config.schema import DatasetConfig
from medseg.data.datasets.base import SampleTransform
from medseg.data.datasets.binary_folder import BinaryFolderSegmentationDataset


class ISIC2016Dataset(BinaryFolderSegmentationDataset):
    """Dataset placeholder for the ISIC 2016 lesion segmentation benchmark."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transform: SampleTransform | None = None,
    ) -> None:
        super().__init__(config=config, split=split, transform=transform)
