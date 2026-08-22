"""Lightweight Stage 2 contract and graph tests."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import compose_config  # noqa: E402
from medseg.data.augmentations import build_image_transform, build_transforms  # noqa: E402
from medseg.data.dataloaders.collate import image_collate, segmentation_collate  # noqa: E402
from medseg.data.datasets import ISIC2017ManifestDataset  # noqa: E402
from medseg.losses import BCELoss  # noqa: E402
from medseg.models import build_model  # noqa: E402
from medseg.models.extensions import ConvLoRA, parameter_counts  # noqa: E402
from medseg.models.heads import EarlySegmentationHead  # noqa: E402
from medseg.training import (  # noqa: E402
    evaluate_consistency,
    load_model_state,
    prepare_adaptation_model,
    split_adaptation_dataset,
    target_adaptation_step,
)


class _ToyAdaptationModel(torch.nn.Module):
    """Tiny model for verifying post-update checkpoint selection."""

    def __init__(self, scale: float = -0.1) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.scale * inputs}

    def encode(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"down3": inputs}


class _ToyESH(torch.nn.Module):
    """Fixed ESH that makes the toy consistency score change after an update."""

    level = 3

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


class Stage2Tests(unittest.TestCase):
    """Verify only the small executable contracts needed before real runs."""

    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_frozen_manifests_have_expected_rows_and_columns(self) -> None:
        expected = {
            "splits/isic2017_duplicate_audit.csv": (
                746,
                {"target_image_id", "target_filename", "source_image_id", "source_filename", "sha256", "reason"},
            ),
            "splits/isic2017_target_adaptation.csv": (
                1003,
                {"image_id", "image_path", "mask_available", "mask_path", "split", "target_labels_allowed_for_training", "seed"},
            ),
            "splits/isic2017_target_eval.csv": (
                251,
                {"image_id", "image_path", "mask_available", "mask_path", "split", "target_labels_allowed_for_training", "seed"},
            ),
        }
        for name, (row_count, columns) in expected.items():
            with self.subTest(name=name):
                with (PROJECT_ROOT / name).open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                self.assertEqual(len(rows), row_count)
                self.assertTrue(columns.issubset(set(reader.fieldnames or [])))
                if "target_labels_allowed_for_training" in columns:
                    self.assertTrue(all(row["target_labels_allowed_for_training"].lower() == "false" for row in rows))

    def test_source_builder_has_no_lora_and_full_adapter_scope_has_expected_count(self) -> None:
        source_config = compose_config(overrides=["model=unet2d_source"])
        source_model = build_model(source_config)
        self.assertFalse(any(isinstance(module, ConvLoRA) for module in source_model.modules()))

        adapter_config = compose_config(overrides=["model=unet2d_convlora", "stage2=convlora_isic2016"])
        adapter_model = build_model(adapter_config)
        names = [name for name, module in adapter_model.named_modules() if isinstance(module, ConvLoRA)]
        self.assertTrue(names)
        self.assertTrue(all(name.split(".")[0] in {"init_path", "down1", "down2", "down3"} for name in names))
        self.assertEqual(parameter_counts(adapter_model)["trainable"], 54870)

    def test_adaptation_graph_freezes_base_and_keeps_soft_pseudo_labels(self) -> None:
        config = compose_config(overrides=["model=unet2d_convlora", "stage2=convlora_isic2016"])
        config.model.n_filters_init = 2
        model = prepare_adaptation_model(config)
        esh = EarlySegmentationHead(in_channels=16, out_channels=1, level=3)
        for parameter in esh.parameters():
            parameter.requires_grad = False
        esh.eval()
        images = torch.rand(1, 3, 32, 32)
        before = model.down1[0].running_mean.detach().clone()
        loss, pseudo, _, adapted_logits = target_adaptation_step(
            model, esh, images, BCELoss(), config=config
        )
        self.assertTrue(pseudo.requires_grad)
        self.assertEqual(adapted_logits.shape, (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" in name))
        self.assertFalse(any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name))
        self.assertFalse(any(parameter.requires_grad for parameter in esh.parameters()))
        self.assertFalse(torch.equal(before, model.down1[0].running_mean))

    def test_pseudo_label_detachment_is_configuration_controlled(self) -> None:
        config = compose_config(overrides=["model=unet2d_convlora", "stage2=convlora_isic2016"])
        config.model.n_filters_init = 2
        model = prepare_adaptation_model(config)
        esh = EarlySegmentationHead(in_channels=16, out_channels=1, level=3)
        for parameter in esh.parameters():
            parameter.requires_grad = False
        esh.eval()
        images = torch.rand(1, 3, 32, 32)

        _, attached, _, _ = target_adaptation_step(
            model, esh, images, BCELoss(), config=config
        )
        self.assertFalse(config.stage2.detach_pseudo_labels)
        self.assertTrue(attached.requires_grad)
        self.assertIsNotNone(attached.grad_fn)

        config.stage2.detach_pseudo_labels = True
        _, detached, _, _ = target_adaptation_step(
            model, esh, images, BCELoss(), config=config
        )
        self.assertFalse(detached.requires_grad)
        self.assertIsNone(detached.grad_fn)
        self.assertTrue(
            any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters())
        )
        self.assertFalse(
            any("lora_" not in name and parameter.requires_grad for name, parameter in model.named_parameters())
        )

    def test_target_manifest_adaptation_is_image_only_and_eval_has_masks(self) -> None:
        config = compose_config(overrides=["stage2=convlora_isic2016"])
        config.preprocessing.size.height = 32
        config.preprocessing.size.width = 32
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            rows = []
            for index in range(2):
                image_id = f"ISIC_2017_{index:03d}"
                Image.fromarray(np.full((12, 16, 3), index * 80 + 20, dtype=np.uint8)).save(images / f"{image_id}.jpg")
                Image.fromarray(np.where(np.indices((12, 16))[0] > 5, 255, 0).astype(np.uint8)).save(masks / f"{image_id}_Segmentation.png")
                rows.append({
                    "image_id": image_id,
                    "image_path": str(images / f"{image_id}.jpg"),
                    "mask_available": "true",
                    "mask_path": str(masks / f"{image_id}_Segmentation.png"),
                    "split": "target_eval",
                    "target_labels_allowed_for_training": "false",
                    "seed": "42",
                })
            manifest = root / "target.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            adaptation = ISIC2017ManifestDataset(
                config,
                manifest,
                images,
                masks,
                transform=build_image_transform(config, "eval"),
                include_mask=False,
            )
            adaptation_batch = next(iter(DataLoader(adaptation, batch_size=2, collate_fn=image_collate)))
            self.assertNotIn("mask", adaptation_batch)
            self.assertEqual(tuple(adaptation_batch["image"].shape), (2, 3, 32, 32))

            evaluation = ISIC2017ManifestDataset(
                config,
                manifest,
                images,
                masks,
                transform=build_transforms(config, "eval"),
                include_mask=True,
            )
            evaluation_batch = next(iter(DataLoader(evaluation, batch_size=2, collate_fn=segmentation_collate)))
            self.assertEqual(tuple(evaluation_batch["image"].shape), (2, 3, 32, 32))
            self.assertEqual(tuple(evaluation_batch["mask"].shape), (2, 1, 32, 32))

    def test_checkpoint_wrapper_and_raw_state_loading(self) -> None:
        config = compose_config(overrides=["model=unet2d_source"])
        model = build_model(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, path)
            restored = build_model(config)
            payload = load_model_state(path, restored)
        self.assertEqual(payload["epoch"], 1)

    def test_consistency_split_is_deterministic_and_disjoint(self) -> None:
        samples = [{"index": index} for index in range(1003)]
        first = split_adaptation_dataset(samples, consistency_fraction=0.2, seed=42)
        second = split_adaptation_dataset(samples, consistency_fraction=0.2, seed=42)
        adaptation, consistency, adaptation_indices, consistency_indices = first
        _, _, adaptation_indices_again, consistency_indices_again = second
        self.assertEqual(len(adaptation), 803)
        self.assertEqual(len(consistency), 200)
        self.assertEqual(adaptation_indices, adaptation_indices_again)
        self.assertEqual(consistency_indices, consistency_indices_again)
        self.assertTrue(set(adaptation_indices).isdisjoint(consistency_indices))
        self.assertEqual(set(adaptation_indices) | set(consistency_indices), set(range(1003)))

    def test_checkpoint_selection_uses_post_update_model_and_preserves_bn_stats(self) -> None:
        from medseg.training.stage2 import adapt_model

        config = compose_config(overrides=["stage2=convlora_isic2016"])
        config.stage2.adaptation_epochs = 1
        config.stage2.adaptation_lr = 1.0
        image_samples = [{"image": torch.ones(1, 2, 2), "sample_id": str(index), "metadata": {}} for index in range(2)]
        adaptation_loader = DataLoader(
            [image_samples[0]],
            batch_size=1,
            shuffle=False,
            collate_fn=image_collate,
        )
        consistency_loader = DataLoader(
            [image_samples[1]],
            batch_size=1,
            shuffle=False,
            collate_fn=image_collate,
        )
        model = _ToyAdaptationModel()
        esh = _ToyESH()
        pre_update_score = evaluate_consistency(model, esh, consistency_loader, "cpu")
        pre_update_scale = float(model.scale.detach())

        with tempfile.TemporaryDirectory() as temporary:
            best_path = adapt_model(
                config=config,
                model=model,
                esh=esh,
                dataloader=adaptation_loader,
                loss_fn=BCELoss(),
                device="cpu",
                output_dir=temporary,
                consistency_dataloader=consistency_loader,
            )
            checkpoint = torch.load(best_path, map_location="cpu")

        post_update_score = evaluate_consistency(model, esh, consistency_loader, "cpu")
        saved_scale = float(checkpoint["model_state_dict"]["scale"])
        self.assertNotEqual(pre_update_scale, saved_scale)
        self.assertNotEqual(pre_update_score, post_update_score)
        self.assertAlmostEqual(
            checkpoint["history"][-1]["consistency_dice_global"],
            post_update_score,
        )
        self.assertAlmostEqual(saved_scale, float(model.scale.detach()))

        config.model.n_filters_init = 2
        bn_model = prepare_adaptation_model(config)
        bn_esh = EarlySegmentationHead(in_channels=16, out_channels=1, level=3)
        bn_esh.eval()
        bn_loader = DataLoader(
            [{"image": torch.rand(3, 32, 32), "sample_id": "bn", "metadata": {}}],
            batch_size=1,
            shuffle=False,
            collate_fn=image_collate,
        )
        before = {
            name: buffer.detach().clone()
            for name, buffer in bn_model.named_buffers()
            if "running_" in name
        }
        evaluate_consistency(bn_model, bn_esh, bn_loader, "cpu")
        after = {
            name: buffer.detach().clone()
            for name, buffer in bn_model.named_buffers()
            if "running_" in name
        }
        self.assertTrue(bn_model.training)
        self.assertFalse(bn_esh.training)
        self.assertEqual(before.keys(), after.keys())
        self.assertTrue(all(torch.equal(before[name], after[name]) for name in before))


if __name__ == "__main__":
    unittest.main()
