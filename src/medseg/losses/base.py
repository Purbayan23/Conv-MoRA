"""Base loss contracts and loss factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from medseg.typing import ModelOutput


class BaseLoss(nn.Module, ABC):
    """Base interface for segmentation losses."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, outputs: ModelOutput | torch.Tensor, targets: Any) -> torch.Tensor:
        """Compute a scalar training loss."""


def _extract_logits(outputs: ModelOutput | torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
    raise TypeError("Loss functions expect model outputs to contain a 'logits' tensor.")


def _extract_targets(targets: Any) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets
    if isinstance(targets, Mapping):
        mask = targets.get("mask")
        if mask is None:
            mask = targets.get("target")
        if isinstance(mask, torch.Tensor):
            return mask
    raise TypeError("Loss functions expect targets to be tensors or mappings containing 'mask'.")


def _prepare_binary_target(targets: Any, logits: torch.Tensor) -> torch.Tensor:
    """Convert a binary mask to the same device, dtype, and shape as logits."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Binary segmentation logits must have shape [B, 1, H, W].")

    target = _extract_targets(targets).to(dtype=logits.dtype, device=logits.device)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape != logits.shape:
        raise ValueError(
            f"Binary target shape must match logits shape {tuple(logits.shape)}, "
            f"received {tuple(target.shape)}."
        )
    return target


def build_loss(config: Any) -> BaseLoss:
    """Build the configured loss function."""

    from medseg.config.schema import AppConfig

    loss_config = config.loss if isinstance(config, AppConfig) else config
    name = loss_config.name.lower()

    from medseg.losses.bce import BinaryCrossEntropyWithLogitsLoss
    from medseg.losses.composite import BCEDiceLoss
    from medseg.losses.dice import DiceLoss

    if name in {"bce", "bce_logits", "binary_cross_entropy"}:
        return BinaryCrossEntropyWithLogitsLoss(from_logits=loss_config.from_logits)
    if name in {"dice"}:
        return DiceLoss(from_logits=loss_config.from_logits, smooth=loss_config.smooth)
    if name in {"bce_dice", "combined", "bce+dice"}:
        return BCEDiceLoss(
            bce_weight=loss_config.bce_weight,
            dice_weight=loss_config.dice_weight,
            from_logits=loss_config.from_logits,
            smooth=loss_config.smooth,
        )

    raise ValueError(f"Unknown loss '{loss_config.name}'.")
