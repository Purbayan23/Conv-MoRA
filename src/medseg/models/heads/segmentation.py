"""Placeholder segmentation head implementations."""

from __future__ import annotations

from medseg.config.schema import HeadConfig
from medseg.models.heads.base import BaseSegmentationHead
from medseg.typing import ModelOutput, TensorLike


class BinarySegmentationHead(BaseSegmentationHead):
    """Reserved binary segmentation head for the baseline model."""

    def __init__(self, config: HeadConfig) -> None:
        self.config = config
        raise NotImplementedError(
            "Segmentation head implementation is intentionally deferred in this scaffold."
        )

    def forward(self, inputs: TensorLike) -> ModelOutput:
        raise NotImplementedError(
            "Segmentation head implementation is intentionally deferred in this scaffold."
        )
