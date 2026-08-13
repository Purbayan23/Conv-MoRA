"""Reference-style residual UNet2D used by the ConvLoRA research branch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from medseg.models.base import BaseSegmentationModel
from medseg.models.blocks.residual import PreActivationConv2d, ResidualBlock2d
from medseg.typing import ModelOutput, TensorLike


class UNet2D(BaseSegmentationModel):
    """Author-style residual U-Net with additive encoder skip connections.

    The default research configuration uses three downsampling stages and
    ``n_filters_init=16``, producing channel widths 16, 32, 64, and 128.
    """

    def __init__(
        self,
        n_chans_in: int = 3,
        n_chans_out: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        pooling_size: int = 2,
        n_filters_init: int = 16,
        esh_level: int = 3,
        dropout: bool = False,
        p: float = 0.1,
        return_all_activations: bool = False,
        convlora_enabled: bool = False,
        convlora_rank: int = 2,
        convlora_alpha: int = 2,
        encoder_insertion_scope: Sequence[str] | str | None = None,
    ) -> None:
        super().__init__()
        if n_filters_init <= 0:
            raise ValueError("n_filters_init must be positive.")
        if pooling_size != 2:
            raise ValueError("The reference UNet2D uses pooling_size=2.")

        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out
        self.n_filters_init = n_filters_init
        self.esh_level = esh_level
        self.return_all_activations = return_all_activations
        self.convlora_enabled = convlora_enabled
        self.convlora_rank = convlora_rank
        self.convlora_alpha = convlora_alpha

        n = n_filters_init
        dropout_layer = lambda: nn.Dropout(p) if dropout else nn.Identity()

        self.init_path = nn.Sequential(
            nn.Conv2d(n_chans_in, n, kernel_size, padding=padding, bias=False),
            nn.ReLU(),
            ResidualBlock2d(n, n, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n, n, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n, n, kernel_size=kernel_size, padding=padding),
        )
        self.shortcut0 = nn.Conv2d(n, n, kernel_size=1)

        self.down1 = nn.Sequential(
            nn.BatchNorm2d(n),
            nn.Conv2d(n, n * 2, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer(),
            ResidualBlock2d(n * 2, n * 2, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 2, n * 2, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 2, n * 2, kernel_size=kernel_size, padding=padding),
        )
        self.shortcut1 = nn.Conv2d(n * 2, n * 2, kernel_size=1)

        self.down2 = nn.Sequential(
            nn.BatchNorm2d(n * 2),
            nn.Conv2d(n * 2, n * 4, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer(),
            ResidualBlock2d(n * 4, n * 4, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 4, n * 4, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 4, n * 4, kernel_size=kernel_size, padding=padding),
        )
        self.shortcut2 = nn.Conv2d(n * 4, n * 4, kernel_size=1)

        self.down3 = nn.Sequential(
            nn.BatchNorm2d(n * 4),
            nn.Conv2d(n * 4, n * 8, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer(),
            ResidualBlock2d(n * 8, n * 8, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 8, n * 8, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 8, n * 8, kernel_size=kernel_size, padding=padding),
            dropout_layer(),
        )

        self.up3 = nn.Sequential(
            ResidualBlock2d(n * 8, n * 8, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 8, n * 8, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 8, n * 8, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n * 8),
            nn.ConvTranspose2d(n * 8, n * 4, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer(),
        )

        self.up2 = nn.Sequential(
            ResidualBlock2d(n * 4, n * 4, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 4, n * 4, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 4, n * 4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n * 4),
            nn.ConvTranspose2d(n * 4, n * 2, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer(),
        )

        self.up1 = nn.Sequential(
            ResidualBlock2d(n * 2, n * 2, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 2, n * 2, kernel_size=kernel_size, padding=padding),
            ResidualBlock2d(n * 2, n * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n * 2),
            nn.ConvTranspose2d(n * 2, n, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer(),
        )

        self.out_path = nn.Sequential(
            ResidualBlock2d(n, n, kernel_size=1, padding=0),
            PreActivationConv2d(n, n_chans_out, kernel_size=1, padding=0),
            nn.BatchNorm2d(n_chans_out),
        )

        if convlora_enabled:
            from medseg.models.extensions.convlora import apply_convlora, mark_only_adapter_as_trainable

            apply_convlora(
                self,
                rank=convlora_rank,
                alpha=convlora_alpha,
                scope=encoder_insertion_scope,
            )
            mark_only_adapter_as_trainable(self)

    @classmethod
    def from_config(cls, config: Any) -> "UNet2D":
        """Construct the research model from a model configuration object."""

        return cls(
            n_chans_in=int(getattr(config, "in_channels")),
            n_chans_out=int(getattr(config, "out_channels")),
            n_filters_init=int(getattr(config, "n_filters_init", 16)),
            esh_level=int(getattr(config, "esh_level", 3)),
            convlora_enabled=bool(getattr(config, "convlora_enabled", False)),
            convlora_rank=int(getattr(config, "convlora_rank", 2)),
            convlora_alpha=int(getattr(config, "convlora_alpha", 2)),
            encoder_insertion_scope=getattr(config, "encoder_insertion_scope", None),
        )

    def encode(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named encoder activations for ESH and adaptation stages."""

        x0 = self.init_path(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        return {"init_path": x0, "down1": x1, "down2": x2, "down3": x3}

    def forward(self, inputs: TensorLike) -> ModelOutput:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("UNet2D expects a torch.Tensor input.")

        features = self.encode(inputs)
        x2_up = self.up3(features["down3"])
        x1_up = self.up2(x2_up + self.shortcut2(features["down2"]))
        x0_up = self.up1(x1_up + self.shortcut1(features["down1"]))
        logits = self.out_path(x0_up + self.shortcut0(features["init_path"]))
        output: dict[str, Any] = {"logits": logits}
        if self.return_all_activations:
            output["features"] = features
        return output


__all__ = ["UNet2D"]
