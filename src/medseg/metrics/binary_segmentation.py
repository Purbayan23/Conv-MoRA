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


class PrecisionMetric(BaseMetric):
    """Thresholded positive predictive value for binary segmentation."""

    def __init__(self, threshold: float = 0.5, from_logits: bool = True) -> None:
        self.threshold = threshold
        self.from_logits = from_logits

    def __call__(self, outputs: ModelOutput | torch.Tensor, targets: Any) -> float:
        true_positive, false_positive, _ = _batch_confusion_counts(
            outputs, targets, self.threshold, self.from_logits
        )
        denominator = true_positive + false_positive
        score = torch.where(denominator > 0, true_positive / denominator, torch.zeros_like(denominator))
        return float(score.mean().item())


class RecallMetric(BaseMetric):
    """Thresholded recall, also known as sensitivity, for binary segmentation."""

    def __init__(self, threshold: float = 0.5, from_logits: bool = True) -> None:
        self.threshold = threshold
        self.from_logits = from_logits

    def __call__(self, outputs: ModelOutput | torch.Tensor, targets: Any) -> float:
        true_positive, _, false_negative = _batch_confusion_counts(
            outputs, targets, self.threshold, self.from_logits
        )
        denominator = true_positive + false_negative
        score = torch.where(denominator > 0, true_positive / denominator, torch.zeros_like(denominator))
        return float(score.mean().item())


def _batch_confusion_counts(
    outputs: ModelOutput | torch.Tensor,
    targets: Any,
    threshold: float,
    from_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample true-positive, false-positive, and false-negative counts."""

    logits = _extract_logits(outputs)
    target = _prepare_binary_target(targets, logits) >= 0.5
    probabilities = torch.sigmoid(logits) if from_logits else logits
    prediction = probabilities >= threshold
    dims = tuple(range(1, prediction.ndim))
    true_positive = (prediction & target).sum(dim=dims).float()
    false_positive = (prediction & ~target).sum(dim=dims).float()
    false_negative = (~prediction & target).sum(dim=dims).float()
    return true_positive, false_positive, false_negative


def binary_confusion_counts(
    outputs: ModelOutput | torch.Tensor,
    targets: Any,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> tuple[int, int, int]:
    """Return dataset-accumulable binary confusion counts for one batch."""

    true_positive, false_positive, false_negative = _batch_confusion_counts(
        outputs, targets, threshold, from_logits
    )
    return (
        int(true_positive.sum().item()),
        int(false_positive.sum().item()),
        int(false_negative.sum().item()),
    )


SensitivityMetric = RecallMetric


__all__ = [
    "DiceMetric",
    "IoUMetric",
    "PrecisionMetric",
    "RecallMetric",
    "SensitivityMetric",
    "binary_confusion_counts",
]
