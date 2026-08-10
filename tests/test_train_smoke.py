"""Training scaffold smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import load_typed_config
from medseg.training import Trainer


class TrainingSmokeTests(unittest.TestCase):
    """Verify the trainer can be constructed from config without hard-coded wiring."""

    def test_trainer_from_config_is_model_agnostic(self) -> None:
        config = load_typed_config()
        trainer = Trainer.from_config(config)
        self.assertEqual(trainer.component_summary["dataset"], "isic2016")
        self.assertEqual(trainer.component_summary["dataset_version"], "2016")
        self.assertIn("unet_ronneberger2015", trainer.component_summary["model"])
        self.assertEqual(trainer.component_summary["preprocessing"], "default_binary_preprocessing")
        self.assertEqual(trainer.component_summary["augmentation"], "default_binary_augmentation")
        self.assertEqual(trainer.component_summary["optimizer"], "adam")
        self.assertEqual(trainer.component_summary["checkpoint_load_enabled"], "false")

    def test_fit_raises_until_training_loop_is_implemented(self) -> None:
        config = load_typed_config()
        trainer = Trainer.from_config(config)
        with self.assertRaises(NotImplementedError):
            trainer.fit()
