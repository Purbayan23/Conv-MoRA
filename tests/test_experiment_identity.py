"""Experiment identity and checkpoint policy tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medseg.config import load_typed_config
from medseg.experiment import build_experiment_identity, build_experiment_manifest, create_run_directories


class ExperimentIdentityTests(unittest.TestCase):
    """Verify experiments are identified independently of prior runs."""

    def test_identity_captures_dataset_and_model_independently(self) -> None:
        config = load_typed_config()
        identity = build_experiment_identity(config)
        self.assertEqual(identity.dataset_name, "isic2016")
        self.assertEqual(identity.model_name, "unet_ronneberger2015")
        self.assertEqual(identity.seed, 42)

    def test_manifest_keeps_checkpoint_loading_explicit(self) -> None:
        config = load_typed_config()
        manifest = build_experiment_manifest(config)
        self.assertFalse(manifest["checkpoint"]["load_enabled"])
        self.assertIsNone(manifest["checkpoint"]["load_path"])
        self.assertEqual(manifest["dataset"]["split_definition_name"], "official_isic2016_2016")

    def test_run_directories_are_associated_with_one_dataset_and_seed(self) -> None:
        config = load_typed_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.experiment.output_root = tmpdir
            run_dirs = create_run_directories(config, timestamp="20260810_000000")
            self.assertIn("dataset=isic2016", str(run_dirs.root))
            self.assertIn("seed=42", str(run_dirs.root))
            self.assertTrue((run_dirs.root / config.experiment.manifest_filename).exists())
