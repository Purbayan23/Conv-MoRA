"""Checkpoint path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class CheckpointPaths:
    """Resolved checkpoint-related paths for one experiment run."""

    root: Path
    latest: Path
    best: Path


def build_checkpoint_paths(run_directory: Path) -> CheckpointPaths:
    """Resolve standard checkpoint locations inside a run directory."""

    checkpoint_root = run_directory / "checkpoints"
    return CheckpointPaths(
        root=checkpoint_root,
        latest=checkpoint_root / "latest.pt",
        best=checkpoint_root / "best.pt",
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_val_dice: float,
    history: list[dict[str, float | int]],
) -> None:
    """Save the minimal state required to resume an experiment."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load model state and optionally optimizer state from a checkpoint."""

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
