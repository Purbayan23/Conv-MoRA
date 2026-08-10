"""Model builder tests for the scaffold."""

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
from medseg.models import (
    DEFAULT_BINARY_MODEL_OUTPUT_CONTRACT,
    DEFAULT_MODEL_INPUT_CONTRACT,
    build_model,
    describe_model_request,
    validate_model_input_tensor,
    validate_model_output,
)


@dataclass(frozen=True)
class FakeTensor:
    """Minimal tensor-like object for model contract tests."""

    shape: tuple[int, ...]
    dtype: str


class ModelBuilderTests(unittest.TestCase):
    """Verify the model builder boundary is explicit before implementation."""

    def test_model_request_description_is_stable(self) -> None:
        config = load_typed_config()
        description = describe_model_request(config)
        self.assertIn("unet_ronneberger2015", description)

    def test_model_input_and_output_contracts_are_explicit(self) -> None:
        self.assertEqual(DEFAULT_MODEL_INPUT_CONTRACT.batch_layout, "NCHW")
        self.assertTrue(DEFAULT_BINARY_MODEL_OUTPUT_CONTRACT.returns_logits)
        self.assertFalse(DEFAULT_BINARY_MODEL_OUTPUT_CONTRACT.applies_sigmoid_inside_model)
        validate_model_input_tensor(FakeTensor(shape=(2, 3, 256, 256), dtype="float32"), expected_channels=3)
        validate_model_output({"logits": FakeTensor(shape=(2, 1, 256, 256), dtype="float32")}, expected_batch_size=2)

    def test_unet_builder_raises_clear_not_implemented(self) -> None:
        config = load_typed_config()
        with self.assertRaises(NotImplementedError):
            build_model(config)
