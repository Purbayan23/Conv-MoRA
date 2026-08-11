"""Regression tests for aggregated binary segmentation metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import load_typed_config
from medseg.losses import BCEDiceLoss
from medseg.validation import Evaluator


class FixedLogitModel(nn.Module):
    """Return the first input channel as segmentation logits."""

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": inputs[:, :1]}


class EvaluatorMetricTests(unittest.TestCase):
    """Verify the evaluator uses one consistent TP/FP/FN aggregation."""

    def test_aggregated_metrics_for_known_confusion_counts(self) -> None:
        config = load_typed_config()
        target = torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]],
            dtype=torch.float32,
        )
        prediction = torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]]],
            dtype=torch.float32,
        )
        logits = torch.where(prediction > 0, torch.tensor(10.0), torch.tensor(-10.0))
        image = torch.zeros(3, 1, 12)
        image[0:1] = logits
        sample = {
            "image": image,
            "mask": target,
            "sample_id": "synthetic",
            "metadata": {},
        }

        results = Evaluator.from_config(config).evaluate(
            model=FixedLogitModel(),
            dataloader=DataLoader([sample], batch_size=1),
            loss_fn=BCEDiceLoss(),
            device="cpu",
        )

        self.assertAlmostEqual(results["dice"], 17.0 / 20.0, places=5)
        self.assertAlmostEqual(results["iou"], 9.0 / 12.0, places=5)
        self.assertAlmostEqual(results["precision"], 8.0 / 10.0, places=5)
        self.assertAlmostEqual(results["recall"], 8.0 / 9.0, places=5)

        raw_iou = 8.0 / 11.0
        raw_dice = 16.0 / 19.0
        self.assertAlmostEqual(raw_dice, 2.0 * raw_iou / (1.0 + raw_iou), places=5)
