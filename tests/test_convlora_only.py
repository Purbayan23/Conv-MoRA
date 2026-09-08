"""Focused tests for ConvLoRA adaptation with frozen BN running statistics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
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
    set_batchnorm_eval,
    split_adaptation_dataset,
    target_adaptation_step,
)


def _image_samples(count: int) -> list[dict[str, object]]:
    return [
        {"image": torch.rand(3, 32, 32), "sample_id": str(index), "metadata": {}}
        for index in range(count)
    ]


class ConvLoRAOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def _config(self, n_filters_init: int = 16):
        config = compose_config(
            overrides=[
                "stage2=convlora_only_isic2017",
                "model=unet2d_convlora",
                "experiment=convlora_only_isic2017",
            ]
        )
        config.model.n_filters_init = n_filters_init
        config.stage2.adaptation_epochs = 1
        return config

    def test_configuration_has_exact_adapter_scope_and_trainable_count(self) -> None:
        config = self._config()
        self.assertTrue(config.stage2.freeze_bn_running_stats)
        self.assertFalse(config.stage2.detach_pseudo_labels)
        self.assertEqual(config.stage2.convlora_rank, 2)
        self.assertEqual(config.stage2.convlora_alpha, 2)
        self.assertEqual(
            config.stage2.insertion_scope,
            ["init_path", "down1", "down2", "down3"],
        )

        model = prepare_adaptation_model(config)
        adapters = [module for module in model.modules() if isinstance(module, ConvLoRA)]
        self.assertTrue(adapters)
        self.assertEqual(parameter_counts(model)["trainable"], 54870)
        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if "lora_" in name
            )
        )
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
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                for parameter in module.parameters()
            )
        )
        self.assertTrue(
            all(
                not module.training
                for module in model.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            )
        )

        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.stage2.adaptation_lr,
        )
        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        trainable_parameters = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        }
        self.assertEqual(optimizer_parameters, trainable_parameters)

    def test_adaptation_updates_lora_but_preserves_bn_running_stats(self) -> None:
        config = self._config(n_filters_init=2)
        model = prepare_adaptation_model(config)
        esh = EarlySegmentationHead(in_channels=16, out_channels=1, level=3)
        for parameter in esh.parameters():
            parameter.requires_grad = False
        esh.eval()

        images = _image_samples(2)
        adaptation_loader = DataLoader(
            images[:1], batch_size=1, shuffle=False, collate_fn=image_collate
        )
        consistency_loader = DataLoader(
            images[1:], batch_size=1, shuffle=False, collate_fn=image_collate
        )
        bn_before = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
            if name.endswith("running_mean") or name.endswith("running_var")
        }
        lora_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }

        loss, _, _, _ = target_adaptation_step(
            model, esh, images[0]["image"].unsqueeze(0), BCELoss(), config=config
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if "lora_" in name
            )
        )
        set_batchnorm_eval(model)

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

        bn_after = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
            if name.endswith("running_mean") or name.endswith("running_var")
        }
        self.assertEqual(bn_before.keys(), bn_after.keys())
        for name in bn_before:
            self.assertTrue(torch.equal(bn_before[name], bn_after[name]))
        self.assertTrue(
            all(
                not module.training
                for module in model.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(lora_before[name], parameter)
                for name, parameter in model.named_parameters()
                if "lora_" in name
            )
        )

    def test_internal_target_split_remains_803_and_200_disjoint(self) -> None:
        adaptation, consistency, adaptation_indices, consistency_indices = split_adaptation_dataset(
            list(range(1003)), consistency_fraction=0.2, seed=42
        )
        self.assertEqual(len(adaptation), 803)
        self.assertEqual(len(consistency), 200)
        self.assertTrue(set(adaptation_indices).isdisjoint(consistency_indices))
        self.assertEqual(
            set(adaptation_indices) | set(consistency_indices),
            set(range(1003)),
        )


if __name__ == "__main__":
    unittest.main()
