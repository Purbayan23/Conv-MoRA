"""Base dataset contracts for segmentation experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from medseg.config.schema import DatasetConfig, SplitConfig
from medseg.data.contracts import DEFAULT_SEGMENTATION_SAMPLE_CONTRACT, SegmentationSampleContract
from medseg.data.splits import DatasetSplit, normalize_split, resolve_split_config
from medseg.typing import SegmentationSample

SampleTransform = Callable[[SegmentationSample], SegmentationSample]


class BaseSegmentationDataset(ABC):
    """Common interface expected from all segmentation datasets."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transform: SampleTransform | None = None,
    ) -> None:
        self.config = config
        self.split: DatasetSplit = normalize_split(split)
        self.split_config: SplitConfig = resolve_split_config(config, self.split)
        self.transform = transform
        self.sample_contract: SegmentationSampleContract = DEFAULT_SEGMENTATION_SAMPLE_CONTRACT

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of indexed samples for the split."""

    @abstractmethod
    def __getitem__(self, index: int) -> SegmentationSample:
        """Return one sample following the canonical sample contract."""
