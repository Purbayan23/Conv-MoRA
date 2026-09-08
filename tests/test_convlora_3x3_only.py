"""Focused tests for the 3x3-only ConvLoRA placement ablation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import compose_config  # noqa: E402
from medseg.data.dataloaders.collate import image_collate  # noqa: E402
from medseg.losses import BCELoss  # noqa: E402
from medseg.models.extensions import ConvLoRA, parameter_counts  # noqa: E402
from medseg.models.heads import EarlySegmentationHead  # noqa: E402
from medseg.training import (  # noqa: E402
    adapt_model,
    prepare_adaptation_model,
    split_adaptation_dataset,
    target_adaptation_step,
)


def _config(stage2_name: str, n_filters_init: int = 16):
    config = compose_config(
        overrides=[
            f"stage2={stage2_name}",
            "model=unet2d_convlora",
            "experiment=convlora_3x3_only_isic2017",
        ]
    )
    config.model.n_filters_init = n_filters_init
    config.stage2.adaptation_epochs = 1
    return config


def _image_samples(count: int) -> list[dict[str, object]]:
    return [
        {"image": torch.rand(3, 32, 32), "sample_id": str(index), "metadata": {}}
        for index in range(count)
    ]


class ConvLoRA3x3OnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_only_encoder_3x3_convolutions_receive_adapters(self) -> None:
        config = _config("convlora_3x3_only_isic2017")
        model = prepare_adaptation_model(config)
        adapters = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, ConvLoRA)
        ]

        self.assertTrue(adapters)
        self.assertTrue(
            all(name.split(".")[0] in {"init_path", "down1", "down2", "down3"} for name, _ in adapters)
        )
        self.assertTrue(all(module.kernel_size == (3, 3) for _, module in adapters))
        self.assertTrue(all(module.r == 2 and module.lora_alpha == 2 for _, module in adapters))

        downsampling = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d)
            and module.kernel_size == (2, 2)
            and module.stride == (2, 2)
        ]
        self.assertEqual({name.split(".")[0] for name, _ in downsampling}, {"down1", "down2", "down3"})
        self.assertTrue(all(not isinstance(module, ConvLoRA) for _, module in downsampling))

        decoder_prefixes = {"up1", "up2", "up3", "out_path", "shortcut0", "shortcut1", "shortcut2"}
        self.assertFalse(
            any(name.split(".")[0] in decoder_prefixes for name, _ in adapters)
        )

        calculated_count = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        )
        self.assertEqual(parameter_counts(model)["trainable"], calculated_count)
        self.assertEqual(calculated_count, 52182)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if "lora_" not in name
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in model.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
                for parameter in module.parameters()
            )
        )
        self.assertTrue(config.stage2.freeze_bn_running_stats)

    def test_historical_scope_still_includes_2x2_downsampling_convolutions(self) -> None:
        config = _config("convlora_isic2016", n_filters_init=2)
        self.assertIsNone(config.stage2.convlora_kernel_size)
        model = prepare_adaptation_model(config)
        downsampling_adapters = [
            module
            for name, module in model.named_modules()
            if name.split(".")[0] in {"down1", "down2", "down3"}
            and isinstance(module, ConvLoRA)
            and module.kernel_size == (2, 2)
            and module.stride == (2, 2)
        ]
        self.assertEqual(len(downsampling_adapters), 3)

    def test_frozen_bn_adaptation_keeps_stats_fixed_and_uses_image_only_batches(self) -> None:
        config = _config("convlora_3x3_only_isic2017", n_filters_init=2)
        model = prepare_adaptation_model(config)
        esh = EarlySegmentationHead(in_channels=16, out_channels=1, level=3)
        for parameter in esh.parameters():
            parameter.requires_grad = False
        esh.eval()

        samples = _image_samples(2)
        adaptation_loader = DataLoader(
            samples[:1], batch_size=1, shuffle=False, collate_fn=image_collate
        )
        consistency_loader = DataLoader(
            samples[1:], batch_size=1, shuffle=False, collate_fn=image_collate
        )
        running_before = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
            if name.endswith("running_mean") or name.endswith("running_var")
        }

        loss, _, _, _ = target_adaptation_step(
            model,
            esh,
            samples[0]["image"].unsqueeze(0),
            BCELoss(),
            config=config,
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if "lora_" in name
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            adapt_model(
                config=config,
                model=model,
                esh=esh,
                dataloader=adaptation_loader,
                loss_fn=BCELoss(),
                device="cpu",
                output_dir=temporary,
                consistency_dataloader=consistency_loader,
            )

        running_after = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
            if name.endswith("running_mean") or name.endswith("running_var")
        }
        self.assertEqual(running_before.keys(), running_after.keys())
        for name in running_before:
            self.assertTrue(torch.equal(running_before[name], running_after[name]))
        self.assertTrue(
            all(
                not module.training
                for module in model.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            )
        )

    def test_internal_split_remains_disjoint(self) -> None:
        adaptation, consistency, adaptation_indices, consistency_indices = split_adaptation_dataset(
            list(range(1003)), consistency_fraction=0.2, seed=42
        )
        self.assertEqual(len(adaptation), 803)
        self.assertEqual(len(consistency), 200)
        self.assertTrue(set(adaptation_indices).isdisjoint(consistency_indices))


if __name__ == "__main__":
    unittest.main()
