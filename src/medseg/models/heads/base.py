"""Base contract for segmentation heads."""

from __future__ import annotations

from abc import ABC, abstractmethod

from medseg.typing import ModelOutput, TensorLike


class BaseSegmentationHead(ABC):
    """Base interface for segmentation heads."""

    @abstractmethod
    def forward(self, inputs: TensorLike) -> ModelOutput:
        """Produce segmentation outputs for downstream post-processing."""
