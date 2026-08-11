"""Base metric contracts and metric factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from medseg.typing import ModelOutput


class BaseMetric(ABC):
    """Base interface for evaluation metrics."""

    @abstractmethod
    def __call__(self, outputs: ModelOutput | torch.Tensor | Mapping[str, Any], targets: Any) -> float:
        """Compute one scalar metric value."""


def _extract_logits(outputs: ModelOutput | torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
    raise TypeError("Metrics expect model outputs to contain a 'logits' tensor.")


def _extract_targets(targets: Any) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets
    if isinstance(targets, Mapping):
        mask = targets.get("mask")
        if mask is None:
            mask = targets.get("target")
        if isinstance(mask, torch.Tensor):
            return mask
    raise TypeError("Metrics expect targets to be tensors or mappings containing 'mask'.")


def _prepare_binary_target(targets: Any, logits: torch.Tensor) -> torch.Tensor:
    """Convert a binary mask to the same device and shape as logits."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Binary segmentation logits must have shape [B, 1, H, W].")

    target = _extract_targets(targets).to(device=logits.device)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape != logits.shape:
        raise ValueError(
            f"Binary target shape must match logits shape {tuple(logits.shape)}, "
            f"received {tuple(target.shape)}."
        )
    return target


def build_metrics(config: Any) -> list[BaseMetric]:
    """Build the configured metric set."""

    from medseg.config.schema import AppConfig

    metrics_config = config.metrics if isinstance(config, AppConfig) else config
    from medseg.metrics.binary_segmentation import (
        DiceMetric,
        IoUMetric,
        PrecisionMetric,
        RecallMetric,
    )

    metrics: list[BaseMetric] = []
    for name in metrics_config.names:
        normalized = name.lower()
        if normalized == "dice":
            metrics.append(DiceMetric(threshold=metrics_config.threshold, from_logits=metrics_config.from_logits))
        elif normalized in {"iou", "jaccard"}:
            metrics.append(IoUMetric(threshold=metrics_config.threshold, from_logits=metrics_config.from_logits))
        elif normalized == "precision":
            metrics.append(
                PrecisionMetric(
                    threshold=metrics_config.threshold,
                    from_logits=metrics_config.from_logits,
                )
            )
        elif normalized in {"recall", "sensitivity"}:
            metrics.append(
                RecallMetric(
                    threshold=metrics_config.threshold,
                    from_logits=metrics_config.from_logits,
                )
            )
        else:
            raise ValueError(f"Unknown metric '{name}'.")
    return metrics
