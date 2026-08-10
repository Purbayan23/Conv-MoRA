"""Inference scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from medseg.config.schema import AppConfig


@dataclass
class Predictor:
    """Inference coordinator for segmentation models."""

    config: AppConfig

    @classmethod
    def from_config(cls, config: AppConfig) -> "Predictor":
        return cls(config=config)

    def predict(self) -> None:
        raise NotImplementedError("Inference implementation is intentionally deferred.")
