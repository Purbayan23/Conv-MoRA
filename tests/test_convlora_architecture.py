"""Focused smoke tests for the reference UNet2D and ConvLoRA infrastructure."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import load_typed_config
from medseg.models import build_model
from medseg.models.architectures.unet import UNetRonneberger2015
from medseg.models.architectures.unet2d import UNet2D
from medseg.models.extensions import ConvLoRA, parameter_counts
from medseg.models.heads import EarlySegmentationHead


class ConvLoRAArchitectureTests(unittest.TestCase):
    """Verify the new research branch without touching the baseline branch."""

    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_convlora_preserves_base_and_can_be_disabled(self) -> None:
        base = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        layer = ConvLoRA.from_conv(base, r=2, lora_alpha=2)
        inputs = torch.randn(2, 3, 16, 16)

        with torch.no_grad():
            disabled_output = layer(inputs)
            base_output = base(inputs)
        self.assertTrue(torch.allclose(disabled_output, base_output))
        self.assertFalse(layer.weight.requires_grad)
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)
        self.assertEqual(disabled_output.shape, (2, 4, 16, 16))

        with torch.no_grad():
            layer.lora_B.fill_(0.1)
            enabled_output = layer(inputs)
            expected_weight = layer.weight + layer.delta_weight * (2.0 / 2.0)
            expected_output = F.conv2d(
                inputs,
                expected_weight,
                layer.bias,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
            )
        self.assertTrue(torch.allclose(enabled_output, expected_output))
        self.assertFalse(torch.allclose(enabled_output, base_output))
        layer.set_lora_enabled(False)
        with torch.no_grad():
            self.assertTrue(torch.allclose(layer(inputs), base_output))

    def test_convlora_merge_unmerge_is_numerically_stable(self) -> None:
        base = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        layer = ConvLoRA.from_conv(base, r=2, lora_alpha=2)
        inputs = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            layer.lora_B.fill_(0.1)
            output_unmerged = layer(inputs)

        layer.eval()
        self.assertTrue(layer.merged)
        with torch.no_grad():
            output_merged = layer(inputs)
        self.assertTrue(torch.allclose(output_unmerged, output_merged, rtol=1e-5, atol=1e-6))

        layer.train()
        self.assertFalse(layer.merged)
        with torch.no_grad():
            output_unmerged_again = layer(inputs)
        self.assertTrue(
            torch.allclose(output_unmerged, output_unmerged_again, rtol=1e-5, atol=1e-6)
        )

    def test_unet2d_and_esh_shapes(self) -> None:
        model = UNet2D(n_chans_in=3, n_chans_out=1, n_filters_init=4)
        model.eval()
        inputs = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            output = model(inputs)
            features = model.encode(inputs)
            esh = EarlySegmentationHead(in_channels=32, out_channels=1, level=3)
            esh.eval()
            esh_output = esh(features["down3"])

        self.assertEqual(output["logits"].shape, (2, 1, 64, 64))
        self.assertEqual(features["down3"].shape, (2, 32, 8, 8))
        self.assertEqual(esh_output.shape, (2, 1, 64, 64))

    def test_configured_convlora_model_inserts_encoder_only(self) -> None:
        config = load_typed_config(overrides=["model=unet2d_convlora"])
        model = build_model(config)
        self.assertEqual(model.esh_level, 3)
        model.eval()
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = model(inputs)

        self.assertEqual(output["logits"].shape, (1, 1, 64, 64))
        lora_names = [name for name, module in model.named_modules() if isinstance(module, ConvLoRA)]
        self.assertTrue(lora_names)
        self.assertTrue(all(name.split(".")[0] in {"init_path", "down1", "down2", "down3"} for name in lora_names))
        self.assertFalse(any(name.startswith(("up1", "up2", "up3", "out_path")) for name in lora_names))

        counts = parameter_counts(model)
        self.assertGreater(counts["total"], counts["trainable"])
        self.assertEqual(counts["trainable"], counts["convlora_trainable"])
        self.assertGreater(counts["convlora_trainable"], 0)

    def test_ronneberger_baseline_remains_independent(self) -> None:
        model = UNetRonneberger2015(
            in_channels=3,
            out_channels=1,
            feature_channels=(4, 8, 16, 32, 64),
        )
        output = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(output["logits"].shape, (1, 1, 32, 32))
        self.assertFalse(any(isinstance(module, ConvLoRA) for module in model.modules()))
