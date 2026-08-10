"""Combined BCE and Dice loss."""

from __future__ import annotations

from typing import Any

import torch

from medseg.losses.bce import BinaryCrossEntropyWithLogitsLoss
from medseg.losses.base import BaseLoss
from medseg.losses.dice import DiceLoss
from medseg.typing import ModelOutput


class BCEDiceLoss(BaseLoss):
    """Weighted combination of binary cross-entropy and Dice loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        from_logits: bool = True,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = BinaryCrossEntropyWithLogitsLoss(from_logits=from_logits)
        self.dice_loss = DiceLoss(from_logits=from_logits, smooth=smooth)

    def forward(self, outputs: ModelOutput | torch.Tensor, targets: Any) -> torch.Tensor:
        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice
