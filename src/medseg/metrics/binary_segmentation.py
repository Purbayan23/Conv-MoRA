"""Binary segmentation metrics for logits and binary masks."""

from __future__ import annotations

from typing import Any

import torch

from medseg.metrics.base import BaseMetric, _extract_logits, _prepare_binary_target
from medseg.typing import ModelOutput


class DiceMetric(BaseMetric):
    """Thresholded Dice score for binary segmentation."""

    def __init__(self, threshold: float = 0.5, from_logits: bool = True, smooth: float = 1.0) -> None:
        self.threshold = threshold
        self.from_logits = from_logits
        self.smooth = smooth

    def __call__(self, outputs: ModelOutput | torch.Tensor, targets: Any) -> float:
        logits = _extract_logits(outputs)
        target = _prepare_binary_target(targets, logits) >= 0.5
        probabilities = torch.sigmoid(logits) if self.from_logits else logits
        prediction = probabilities >= self.threshold
        dims = tuple(range(1, prediction.ndim))
        intersection = (prediction & target).sum(dim=dims).float()
        denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
        score = (2.0 * intersection + self.smooth) / (denominator.float() + self.smooth)
        return float(score.mean().item())


class IoUMetric(BaseMetric):
    """Thresholded intersection-over-union score for binary segmentation."""

    def __init__(self, threshold: float = 0.5, from_logits: bool = True, smooth: float = 1.0) -> None:
        self.threshold = threshold
        self.from_logits = from_logits
        self.smooth = smooth

    def __call__(self, outputs: ModelOutput | torch.Tensor, targets: Any) -> float:
        logits = _extract_logits(outputs)
        target = _prepare_binary_target(targets, logits) >= 0.5
        probabilities = torch.sigmoid(logits) if self.from_logits else logits
        prediction = probabilities >= self.threshold
        dims = tuple(range(1, prediction.ndim))
        intersection = (prediction & target).sum(dim=dims).float()
        union = (prediction | target).sum(dim=dims).float()
        score = (intersection + self.smooth) / (union + self.smooth)
        return float(score.mean().item())


__all__ = ["DiceMetric", "IoUMetric"]
