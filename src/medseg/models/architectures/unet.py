"""Placeholder for the original U-Net baseline architecture."""

from __future__ import annotations

from medseg.config.schema import ModelConfig
from medseg.models.base import BaseSegmentationModel
from medseg.typing import ModelOutput, TensorLike


class UNetRonneberger2015(BaseSegmentationModel):
    """Reserved class for the original U-Net baseline."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        raise NotImplementedError("U-Net implementation is intentionally deferred in this scaffold.")

    def forward(self, inputs: TensorLike) -> ModelOutput:
        raise NotImplementedError("U-Net implementation is intentionally deferred in this scaffold.")
