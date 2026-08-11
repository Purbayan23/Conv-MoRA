"""Configuration loading tests for the scaffold."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import compose_config, load_typed_config
from medseg.config.schema import AppConfig


class ConfigLoadingTests(unittest.TestCase):
    """Verify YAML composition and structured config loading."""

    def test_compose_config_uses_expected_baseline(self) -> None:
        config = compose_config()
        self.assertEqual(config.project.name, "conv-mora-seg")
        self.assertEqual(config.dataset.name, "isic2016")
        self.assertEqual(config.model.architecture, "unet_ronneberger2015")
        self.assertEqual(config.preprocessing.name, "default_binary_preprocessing")
        self.assertEqual(config.augmentation.name, "default_binary_augmentation")

    def test_load_typed_config_returns_app_config(self) -> None:
        config = load_typed_config()
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.optimizer.name, "adam")
        self.assertEqual(config.head.name, "binary_segmentation_head")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.split, "val")
        self.assertFalse(config.checkpoint.load_enabled)

    def test_evaluation_split_can_be_overridden(self) -> None:
        config = load_typed_config(overrides=["split=test"])
        self.assertEqual(config.split, "test")
