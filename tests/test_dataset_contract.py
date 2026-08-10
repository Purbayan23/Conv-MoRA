"""Dataset contract tests for the scaffold."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import load_typed_config
from medseg.data import build_dataset, build_model_ready_sample, validate_segmentation_sample
from medseg.data.augmentations.base import IdentityTransform
from medseg.data.contracts import ContractValidationError
from medseg.data.datasets.binary_folder import BinaryFolderSegmentationDataset
from medseg.data.datasets.isic2016 import ISIC2016Dataset


@dataclass(frozen=True)
class FakeTensor:
    """Minimal tensor-like object for contract tests."""

    shape: tuple[int, ...]
    dtype: str
    unique_values: tuple[float, ...] | None = None


class DatasetContractTests(unittest.TestCase):
    """Verify the placeholder dataset follows the intended contract surface."""

    def test_isic_dataset_can_be_built_from_config(self) -> None:
        config = load_typed_config()
        dataset = build_dataset(config, split="train", transform=IdentityTransform())
        self.assertIsInstance(dataset, ISIC2016Dataset)
        self.assertEqual(dataset.split.value, "train")
        self.assertEqual(dataset.split_config.definition_source, "official")
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.sample_contract.required_keys, ("image", "mask", "sample_id", "metadata"))

    def test_common_sample_contract_accepts_model_ready_binary_sample(self) -> None:
        sample = build_model_ready_sample(
            image=FakeTensor(shape=(3, 256, 256), dtype="float32"),
            mask=FakeTensor(shape=(1, 256, 256), dtype="float32", unique_values=(0.0, 1.0)),
            sample_id="ISIC_0001",
            metadata={"dataset_name": "isic2016", "split": "train"},
        )
        validate_segmentation_sample(sample)

    def test_image_and_mask_spatial_dimensions_must_match(self) -> None:
        sample = {
            "image": FakeTensor(shape=(3, 256, 256), dtype="float32"),
            "mask": FakeTensor(shape=(1, 128, 256), dtype="float32", unique_values=(0.0, 1.0)),
            "sample_id": "ISIC_0002",
            "metadata": {"dataset_name": "isic2016"},
        }
        with self.assertRaises(ContractValidationError):
            validate_segmentation_sample(sample)

    def test_masks_must_be_binary_after_preprocessing(self) -> None:
        sample = {
            "image": FakeTensor(shape=(3, 256, 256), dtype="float32"),
            "mask": FakeTensor(shape=(1, 256, 256), dtype="float32", unique_values=(0.0, 0.5, 1.0)),
            "sample_id": "ISIC_0003",
            "metadata": {"dataset_name": "isic2016"},
        }
        with self.assertRaises(ContractValidationError):
            validate_segmentation_sample(sample)

    def test_dataset_adapters_expose_the_same_common_contract(self) -> None:
        config = load_typed_config()
        generic_dataset = BinaryFolderSegmentationDataset(
            config=config.dataset,
            split="val",
            transform=IdentityTransform(),
        )
        isic_dataset = ISIC2016Dataset(
            config=config.dataset,
            split="val",
            transform=IdentityTransform(),
        )
        self.assertEqual(generic_dataset.sample_contract, isic_dataset.sample_contract)
        self.assertEqual(generic_dataset.split_config.name, "val")
