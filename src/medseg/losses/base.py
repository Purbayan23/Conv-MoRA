"""Base loss contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from medseg.config.schema import AppConfig, LossConfig
from medseg.typing import ModelOutput, TensorLike


class BaseLoss(ABC):
    """Base interface for segmentation losses."""

    @abstractmethod
    def __call__(self, outputs: ModelOutput, targets: TensorLike) -> TensorLike:
        """Compute a scalar training loss."""


def build_loss(config: AppConfig | LossConfig) -> BaseLoss:
    """Build the configured loss function."""

    loss_config = config.loss if isinstance(config, AppConfig) else config
    raise NotImplementedError(
        f"Loss '{loss_config.name}' is configured but not implemented in the scaffold yet."
    )
