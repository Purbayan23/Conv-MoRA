"""Validation loop for binary segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from medseg.config.schema import AppConfig
from medseg.losses.base import BaseLoss
from medseg.metrics.base import BaseMetric, build_metrics
from medseg.metrics.binary_segmentation import DiceMetric, IoUMetric


@dataclass
class Evaluator:
    """Model-agnostic validation coordinator."""

    config: AppConfig

    @classmethod
    def from_config(cls, config: AppConfig) -> "Evaluator":
        return cls(config=config)

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: BaseLoss,
        device: torch.device | str,
        metrics: dict[str, BaseMetric] | None = None,
    ) -> dict[str, float]:
        """Evaluate loss and binary segmentation metrics over one dataloader."""

        metric_map = metrics or self._configured_metrics()
        was_training = model.training
        model.eval()
        totals = {"loss": 0.0, **{name: 0.0 for name in metric_map}}
        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                batch_size = images.shape[0]
                totals["loss"] += float(loss.item()) * batch_size
                for name, metric in metric_map.items():
                    totals[name] += float(metric(outputs, masks)) * batch_size
                sample_count += batch_size

        if was_training:
            model.train()
        if sample_count == 0:
            raise ValueError("Cannot evaluate an empty dataloader.")
        return {name: value / sample_count for name, value in totals.items()}

    def _configured_metrics(self) -> dict[str, BaseMetric]:
        configured = build_metrics(self.config)
        metric_map: dict[str, BaseMetric] = {}
        for name, metric in zip(self.config.metrics.names, configured):
            normalized = name.lower()
            if normalized == "dice" and isinstance(metric, DiceMetric):
                metric_map["dice"] = metric
            elif normalized in {"iou", "jaccard"} and isinstance(metric, IoUMetric):
                metric_map["iou"] = metric
        if "dice" not in metric_map:
            metric_map["dice"] = DiceMetric(
                threshold=self.config.metrics.threshold,
                from_logits=self.config.metrics.from_logits,
            )
        if "iou" not in metric_map:
            metric_map["iou"] = IoUMetric(
                threshold=self.config.metrics.threshold,
                from_logits=self.config.metrics.from_logits,
            )
        return metric_map
