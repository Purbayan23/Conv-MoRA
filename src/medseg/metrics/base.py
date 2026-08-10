"""Base metric contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from medseg.config.schema import AppConfig, MetricsConfig
from medseg.typing import ModelOutput, TensorLike


class BaseMetric(ABC):
    """Base interface for evaluation metrics."""

    @abstractmethod
    def __call__(self, outputs: ModelOutput, targets: TensorLike) -> float:
        """Compute one scalar metric value."""


def build_metrics(config: AppConfig | MetricsConfig) -> list[BaseMetric]:
    """Build the configured metric set."""

    metrics_config = config.metrics if isinstance(config, AppConfig) else config
    raise NotImplementedError(
        "Metrics are configured but not implemented in the scaffold yet: "
        + ", ".join(metrics_config.names)
    )
