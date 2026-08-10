"""Binary cross entropy with logits loss."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from medseg.losses.base import BaseLoss, _extract_logits, _prepare_binary_target
from medseg.typing import ModelOutput


class BinaryCrossEntropyWithLogitsLoss(BaseLoss):
    """Binary cross-entropy loss for segmentation logits."""

    def __init__(self, from_logits: bool = True) -> None:
        super().__init__()
        self.from_logits = from_logits

    def forward(self, outputs: ModelOutput | torch.Tensor, targets: Any) -> torch.Tensor:
        logits = _extract_logits(outputs)
        target_tensor = _prepare_binary_target(targets, logits)
        if not self.from_logits:
            probabilities = logits.clamp(1e-7, 1.0 - 1e-7)
            return F.binary_cross_entropy(probabilities, target_tensor)
        return F.binary_cross_entropy_with_logits(logits, target_tensor)
