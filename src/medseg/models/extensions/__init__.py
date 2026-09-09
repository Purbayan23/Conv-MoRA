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
from medseg.models.extensions.convmora import (
    ConvMoRA,
    ConvMoRAAdapter,
    apply_convmora,
    mark_only_convmora_as_trainable,
)

__all__ = [
    "Adapter",
    "ConvLoRA",
    "ConvLoRAAdapter",
    "ConvMoRA",
    "ConvMoRAAdapter",
    "ModelExtension",
    "apply_convlora",
    "apply_convmora",
    "freeze_base_model",
    "mark_only_adapter_as_trainable",
    "mark_only_convmora_as_trainable",
    "parameter_counts",
]
