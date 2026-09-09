"""Stage-2 integration tests for the ConvMoRA frozen-BN mode."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import compose_config  # noqa: E402
from medseg.models.extensions import ConvLoRA, ConvMoRA  # noqa: E402
from medseg.training.stage2 import (  # noqa: E402
    build_base_unet,
    prepare_adaptation_model,
    set_batchnorm_eval,
)


class ConvMoRAIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    @staticmethod
    def _config():
        config = compose_config(
            overrides=[
                "stage2=convmora_3x3_only_isic2017",
                "model=unet2d_convlora",
                "experiment=convmora_3x3_only_isic2017",
            ]
        )
        return config

    def test_configuration_and_integrated_scope(self) -> None:
        config = self._config()
        self.assertEqual(config.stage2.adaptation_mode, "convmora_3x3_only_frozen_bn")
        self.assertTrue(config.stage2.freeze_bn_running_stats)
        self.assertEqual(config.stage2.convlora_rank, 2)
        self.assertEqual(config.stage2.convlora_kernel_size, 3)

        model = prepare_adaptation_model(config)
        self.assertFalse(any(isinstance(module, ConvLoRA) for module in model.modules()))
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

        shape_counts = Counter(
            (module.in_channels, module.out_channels, module.m)
            for _, module in adapters
        )
        self.assertEqual(
            shape_counts,
            Counter(
                {
                    (3, 16, 18): 1,
                    (16, 16, 24): 6,
                    (32, 32, 34): 6,
                    (64, 64, 48): 6,
                    (128, 128, 68): 6,
                }
            ),
        )

        downsampling = [
            module
            for module in model.modules()
            if isinstance(module, nn.Conv2d)
            and module.kernel_size == (2, 2)
            and module.stride == (2, 2)
        ]
        self.assertTrue(downsampling)
        self.assertTrue(all(not isinstance(module, ConvMoRA) for module in downsampling))
        self.assertFalse(
            any(
                isinstance(module, ConvMoRA)
                for name, module in model.named_modules()
                if name.split(".")[0] in {"up1", "up2", "up3", "out_path"}
            )
        )

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(sum(parameter.numel() for parameter in trainable), 52284)
        adapter_parameter_ids = {
            id(module.M) for module in model.modules() if isinstance(module, ConvMoRA)
        }
        self.assertEqual({id(parameter) for parameter in trainable}, adapter_parameter_ids)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in model.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
                for parameter in module.parameters()
            )
        )
        model.train()
        set_batchnorm_eval(model)
        self.assertTrue(
            all(
                not module.training
                for module in model.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            )
        )

    def test_zero_adapter_forward_matches_source_model(self) -> None:
        config = self._config()
        source_model = build_base_unet(config)
        integrated_model = prepare_adaptation_model(config)

        integrated_model.load_state_dict(source_model.state_dict(), strict=False)
        source_model.eval()
        integrated_model.eval()
        inputs = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            source_output = source_model(inputs)["logits"]
            integrated_output = integrated_model(inputs)["logits"]
        self.assertEqual(integrated_output.shape, (1, 1, 256, 256))
        self.assertTrue(torch.allclose(integrated_output, source_output, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
