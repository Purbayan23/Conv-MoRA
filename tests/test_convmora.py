"""Focused tests for the standalone Type-6 ConvMoRA module."""

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

from medseg.models.architectures.unet2d import UNet2D  # noqa: E402
from medseg.models.extensions import (  # noqa: E402
    ConvMoRA,
    apply_convmora,
    mark_only_convmora_as_trainable,
)


EXPECTED_SHAPES = (
    (3, 16, 3, 18),
    (16, 16, 3, 24),
    (32, 32, 3, 34),
    (64, 64, 3, 48),
    (128, 128, 3, 68),
)


class ConvMoRATests(unittest.TestCase):
    @staticmethod
    def _layer(in_channels: int, out_channels: int, kernel_size: int) -> ConvMoRA:
        base = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        return ConvMoRA.from_conv(base, r_conv=2)

    @staticmethod
    def _set_deterministic_matrix(layer: ConvMoRA) -> None:
        values = torch.arange(layer.m * layer.m, dtype=layer.M.dtype).reshape_as(layer.M)
        with torch.no_grad():
            layer.M.copy_((values + 1.0) / (layer.m * layer.m))

    def test_parameter_shapes_and_zero_initialization(self) -> None:
        for in_channels, out_channels, kernel_size, expected_m in EXPECTED_SHAPES:
            layer = self._layer(in_channels, out_channels, kernel_size)
            self.assertEqual(layer.M.shape, (expected_m, expected_m))
            self.assertEqual(layer.m, expected_m)
            self.assertTrue(torch.equal(layer.M, torch.zeros_like(layer.M)))
            self.assertTrue(layer.M.requires_grad)
            self.assertFalse(layer.weight.requires_grad)
            self.assertIsNotNone(layer.bias)
            self.assertFalse(layer.bias.requires_grad)

    def test_complete_3x3_only_model_parameter_count_and_scope(self) -> None:
        model = UNet2D(n_chans_in=3, n_chans_out=1, n_filters_init=16)
        apply_convmora(model, r_conv=2, scope=None, kernel_size=3)
        mark_only_convmora_as_trainable(model)

        adapters = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, ConvMoRA)
        ]
        self.assertEqual(len(adapters), 25)
        self.assertTrue(
            all(name.split(".")[0] in {"init_path", "down1", "down2", "down3"} for name, _ in adapters)
        )
        self.assertTrue(all(module.kernel_size == (3, 3) for _, module in adapters))

        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertEqual(trainable_count, 52284)
        adapter_parameter_ids = {
            id(module.M) for module in model.modules() if isinstance(module, ConvMoRA)
        }
        trainable_parameter_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        self.assertEqual(trainable_parameter_ids, adapter_parameter_ids)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.parameters()
                if id(parameter) not in adapter_parameter_ids
            )
        )

    def test_zero_delta_and_source_equivalent_forward(self) -> None:
        for in_channels, out_channels, kernel_size, _ in EXPECTED_SHAPES:
            base = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=True,
            )
            layer = ConvMoRA.from_conv(base, r_conv=2)
            inputs = torch.randn(2, in_channels, 13, 13)

            expected = base(inputs)
            actual = layer(inputs)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
            self.assertTrue(torch.allclose(layer.delta_weight_tilde, torch.zeros_like(layer.delta_weight_tilde)))
            self.assertTrue(torch.allclose(layer.delta_weight, torch.zeros_like(layer.delta_weight)))

    def test_nonzero_update_shape_and_forward_merged_equivalence(self) -> None:
        for in_channels, out_channels, kernel_size, _ in EXPECTED_SHAPES:
            base = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=True,
            )
            layer = ConvMoRA.from_conv(base, r_conv=2)
            self._set_deterministic_matrix(layer)
            inputs = torch.randn(2, in_channels, 11, 11)

            layer.train()
            adapter_output = layer(inputs)
            delta = layer.delta_weight
            merged_weight = layer.weight + delta
            reference_output = F.conv2d(
                inputs,
                merged_weight,
                layer.bias,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
            )
            self.assertEqual(delta.shape, base.weight.shape)
            self.assertTrue(torch.allclose(adapter_output, reference_output, atol=1e-5, rtol=1e-5))

    def test_merge_unmerge_and_stem_prefix_repeat(self) -> None:
        base = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        layer = ConvMoRA.from_conv(base, r_conv=2)
        self.assertEqual(layer.input_dim, 9)
        self.assertEqual(layer.m, 18)
        self.assertLess(layer.input_dim, layer.m)
        self._set_deterministic_matrix(layer)
        inputs = torch.randn(2, 3, 17, 17)

        layer.train()
        original_weight = layer.weight.detach().clone()
        unmerged_output = layer(inputs)
        layer.eval()
        merged_output = layer(inputs)
        self.assertTrue(layer.merged)
        self.assertTrue(torch.allclose(unmerged_output, merged_output, atol=1e-5, rtol=1e-5))

        layer.train()
        unmerged_again = layer(inputs)
        self.assertFalse(layer.merged)
        self.assertTrue(torch.allclose(layer.weight, original_weight, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(unmerged_output, unmerged_again, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
