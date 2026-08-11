"""Validation loop for binary segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from medseg.config.schema import AppConfig
from medseg.losses.base import BaseLoss
from medseg.metrics.base import BaseMetric, build_metrics
from medseg.metrics.binary_segmentation import (
    DiceMetric,
    IoUMetric,
    PrecisionMetric,
    RecallMetric,
    binary_confusion_counts,
)
from medseg.visualization.overlays import save_qualitative_examples


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
        qualitative_dir: Path | None = None,
        qualitative_examples: int = 0,
    ) -> dict[str, float]:
        """Evaluate loss and aggregate binary segmentation metrics over a dataloader."""

        metric_map = metrics or self._configured_metrics()
        threshold, from_logits, smooth = self._metric_settings(metric_map)
        was_training = model.training
        model.eval()
        total_loss = 0.0
        sample_count = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        qualitative_saved = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                batch_size = images.shape[0]
                total_loss += float(loss.item()) * batch_size
                batch_tp, batch_fp, batch_fn = binary_confusion_counts(
                    outputs,
                    masks,
                    threshold=threshold,
                    from_logits=from_logits,
                )
                true_positive += batch_tp
                false_positive += batch_fp
                false_negative += batch_fn
                if qualitative_dir is not None and qualitative_saved < qualitative_examples:
                    qualitative_saved += save_qualitative_examples(
                        images=images,
                        masks=masks,
                        logits=outputs["logits"],
                        sample_ids=batch.get("sample_id"),
                        output_dir=qualitative_dir,
                        threshold=threshold,
                        start_index=qualitative_saved,
                        max_examples=qualitative_examples,
                    )
                sample_count += batch_size

        if was_training:
            model.train()
        if sample_count == 0:
            raise ValueError("Cannot evaluate an empty dataloader.")
        union = true_positive + false_positive + false_negative
        return {
            "loss": total_loss / sample_count,
            "dice": (2.0 * true_positive + smooth) / (2.0 * true_positive + union + smooth),
            "iou": (true_positive + smooth) / (union + smooth),
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive > 0
                else 0.0
            ),
            "recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative > 0
                else 0.0
            ),
        }

    def _configured_metrics(self) -> dict[str, BaseMetric]:
        configured = build_metrics(self.config)
        metric_map: dict[str, BaseMetric] = {}
        for name, metric in zip(self.config.metrics.names, configured):
            normalized = name.lower()
            if normalized == "dice" and isinstance(metric, DiceMetric):
                metric_map["dice"] = metric
            elif normalized in {"iou", "jaccard"} and isinstance(metric, IoUMetric):
                metric_map["iou"] = metric
            elif normalized == "precision" and isinstance(metric, PrecisionMetric):
                metric_map["precision"] = metric
            elif normalized in {"recall", "sensitivity"} and isinstance(metric, RecallMetric):
                metric_map["recall"] = metric
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
        if "precision" not in metric_map:
            metric_map["precision"] = PrecisionMetric(
                threshold=self.config.metrics.threshold,
                from_logits=self.config.metrics.from_logits,
            )
        if "recall" not in metric_map:
            metric_map["recall"] = RecallMetric(
                threshold=self.config.metrics.threshold,
                from_logits=self.config.metrics.from_logits,
            )
        return metric_map

    def _metric_settings(
        self,
        metrics: dict[str, BaseMetric],
    ) -> tuple[float, bool, float]:
        """Resolve thresholding settings from the configured metric objects."""

        first_metric = next(iter(metrics.values()))
        threshold = float(getattr(first_metric, "threshold", self.config.metrics.threshold))
        from_logits = bool(getattr(first_metric, "from_logits", self.config.metrics.from_logits))
        dice_metric = metrics.get("dice")
        smooth = float(getattr(dice_metric, "smooth", self.config.metrics.smooth))
        return threshold, from_logits, smooth
