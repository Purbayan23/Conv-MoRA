"""Focused tests for the BN-statistics-only target ablation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import compose_config  # noqa: E402
from medseg.data.dataloaders.collate import image_collate, segmentation_collate  # noqa: E402
from medseg.losses import BCELoss  # noqa: E402
from medseg.models.extensions import ConvLoRA  # noqa: E402
from medseg.training import (  # noqa: E402
    collect_bn_statistics,
    prepare_bn_only_model,
    run_bn_only,
)


class _ImageOnlyDataset(Dataset[dict[str, object]]):
    """Dataset contract that deliberately exposes no mask during adaptation."""

    def __init__(self, count: int = 2) -> None:
        self.images = [torch.rand(3, 32, 32) for _ in range(count)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": self.images[index],
            "sample_id": str(index),
            "metadata": {},
        }


class _EvaluationDataset(Dataset[dict[str, object]]):
    """Small labeled dataset used only by post-pass evaluation."""

    def __init__(self, count: int = 2) -> None:
        self.images = [torch.rand(3, 32, 32) for _ in range(count)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": self.images[index],
            "mask": torch.zeros(1, 32, 32),
            "sample_id": str(index),
            "metadata": {},
        }


class BNOnlyTests(unittest.TestCase):
    def _config(self):
        config = compose_config(
            overrides=[
                "stage2=bn_only_isic2017",
                "model=unet2d_source",
                "experiment=bn_only_isic2017",
            ]
        )
        config.model.n_filters_init = 2
        config.stage2.adaptation_epochs = 1
        self.assertEqual(config.stage2.adaptation_mode, "bn_only")
        self.assertFalse(config.model.convlora_enabled)
        return config

    def test_bn_only_freezes_parameters_and_updates_only_running_stats(self) -> None:
        config = self._config()
        model = prepare_bn_only_model(config)

        self.assertFalse(any(isinstance(module, ConvLoRA) for module in model.modules()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        parameters_before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        buffers_before = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
            if "running_" in name
        }

        adaptation_loader = DataLoader(
            _ImageOnlyDataset(),
            batch_size=2,
            shuffle=False,
            collate_fn=image_collate,
        )
        collect_bn_statistics(model, adaptation_loader, "cpu")

        self.assertTrue(
            any(
                not torch.equal(buffers_before[name], buffer)
                for name, buffer in model.named_buffers()
                if name in buffers_before
            )
        )
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameters_before[name], parameter))

    def test_bn_only_records_reference_and_final_pass_without_adaptation_masks(self) -> None:
        config = self._config()
        model = prepare_bn_only_model(config)
        adaptation_loader = DataLoader(
            _ImageOnlyDataset(),
            batch_size=2,
            shuffle=False,
            collate_fn=image_collate,
        )
        evaluation_loader = DataLoader(
            _EvaluationDataset(),
            batch_size=2,
            shuffle=False,
            collate_fn=segmentation_collate,
        )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = run_bn_only(
                config=config,
                model=model,
                adaptation_dataloader=adaptation_loader,
                evaluation_dataloader=evaluation_loader,
                loss_fn=BCELoss(),
                device="cpu",
                output_dir=temporary,
                consistency_subset_size=1,
            )
            with (Path(temporary) / "history.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            payload = torch.load(checkpoint, map_location="cpu")

        self.assertEqual(summary["adaptation_mode"], "bn_only")
        self.assertEqual([record["epoch"] for record in summary["history"]], [0, 1])
        self.assertEqual(payload["adaptation_mode"], "bn_only")
        self.assertEqual(payload["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
