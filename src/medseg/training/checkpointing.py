"""Checkpoint path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
