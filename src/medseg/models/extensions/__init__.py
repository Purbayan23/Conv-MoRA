"""Extension hook exports."""

from medseg.models.extensions.base import Adapter, ModelExtension
from medseg.models.extensions.convlora import (
    ConvLoRA,
    ConvLoRAAdapter,
    apply_convlora,
    freeze_base_model,
    mark_only_adapter_as_trainable,
    parameter_counts,
)

__all__ = [
    "Adapter",
    "ConvLoRA",
    "ConvLoRAAdapter",
    "ModelExtension",
    "apply_convlora",
    "freeze_base_model",
    "mark_only_adapter_as_trainable",
    "parameter_counts",
]
