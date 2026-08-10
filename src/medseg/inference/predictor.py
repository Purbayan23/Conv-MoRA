"""Minimal inference helper for segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from medseg.config.schema import AppConfig
from medseg.inference.postprocess import resolve_binary_threshold
from medseg.training.checkpointing import load_checkpoint


@dataclass
class Predictor:
    """Inference coordinator for segmentation models."""

    config: AppConfig
    model: nn.Module | None = None
    device: torch.device | str = "cpu"

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        model: nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> "Predictor":
        return cls(config=config, model=model, device=device)

    def load(self, checkpoint_path: str) -> dict[str, Any]:
        """Load model weights for inference."""

        if self.model is None:
            raise ValueError("A model must be supplied before loading an inference checkpoint.")
        return load_checkpoint(Path(checkpoint_path), self.model, device=self.device)

    def predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits, probabilities, and thresholded predictions for a batch."""

        if self.model is None:
            raise ValueError("A model must be supplied before prediction.")
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            logits = outputs["logits"]
            probabilities = torch.sigmoid(logits)
            threshold = resolve_binary_threshold(self.config.head.threshold)
            predictions = (probabilities >= threshold).to(dtype=torch.float32)
        if was_training:
            self.model.train()
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }
