"""Transform and preprocessing contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import load_typed_config
from medseg.data import categorical_values_preserved, mask_values_are_binary
from medseg.data.augmentations import describe_transform_plan


class TransformContractTests(unittest.TestCase):
    """Verify preprocessing and augmentation policies stay safe for segmentation."""

    def test_train_and_eval_transform_policies_are_separate(self) -> None:
        config = load_typed_config()
        train_plan = describe_transform_plan(config, stage="train")
        eval_plan = describe_transform_plan(config, stage="eval")
        self.assertTrue(train_plan.allow_random)
        self.assertFalse(eval_plan.allow_random)
        self.assertGreater(len(train_plan.spatial_transforms), 0)
        self.assertEqual(eval_plan.spatial_transforms, ())

    def test_mask_interpolation_policy_is_nearest(self) -> None:
        config = load_typed_config()
        self.assertEqual(config.preprocessing.interpolation.mask, "nearest")
        self.assertEqual(config.augmentation.policy.mask_interpolation, "nearest")

    def test_categorical_mask_values_are_preserved_by_contract(self) -> None:
        self.assertTrue(mask_values_are_binary((0, 1, 1, 0)))
        self.assertTrue(categorical_values_preserved((0, 1), (0, 1)))
        self.assertFalse(categorical_values_preserved((0, 1), (0, 0.5, 1)))
