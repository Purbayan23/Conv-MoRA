"""Generic binary segmentation dataset scaffold."""

from __future__ import annotations

from medseg.config.schema import DatasetConfig
from medseg.data.datasets.base import BaseSegmentationDataset, SampleTransform
from medseg.typing import SegmentationSample


class BinaryFolderSegmentationDataset(BaseSegmentationDataset):
    """Folder- or manifest-based dataset scaffold for binary segmentation."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transform: SampleTransform | None = None,
    ) -> None:
        super().__init__(config=config, split=split, transform=transform)
        self.records: list[str] = []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SegmentationSample:
        raise IndexError(
            "No dataset records have been indexed yet. Dataset parsing is intentionally deferred."
        )
