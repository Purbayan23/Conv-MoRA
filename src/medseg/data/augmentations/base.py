"""Base transform contracts for segmentation samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from medseg.typing import SegmentationSample


class Transform(Protocol):
    """Protocol for transforms that preserve the sample contract."""

    def __call__(self, sample: SegmentationSample) -> SegmentationSample:
        """Transform a dataset sample."""


@dataclass(frozen=True)
class IdentityTransform:
    """No-op transform used until the augmentation pipeline is implemented."""

    def __call__(self, sample: SegmentationSample) -> SegmentationSample:
        return sample
