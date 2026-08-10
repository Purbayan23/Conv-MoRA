"""Dice loss for binary segmentation."""

from __future__ import annotations

from typing import Any

import torch

from medseg.losses.base import BaseLoss, _extract_logits, _prepare_binary_target
from medseg.typing import ModelOutput


class DiceLoss(BaseLoss):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, from_logits: bool = True, smooth: float = 1.0) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth

    def forward(self, outputs: ModelOutput | torch.Tensor, targets: Any) -> torch.Tensor:
        logits = _extract_logits(outputs)
        target_tensor = _prepare_binary_target(targets, logits)
        probabilities = torch.sigmoid(logits) if self.from_logits else logits
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * target_tensor).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target_tensor.sum(dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()
