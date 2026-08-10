"""Experiment directory management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from medseg.config.schema import AppConfig
from medseg.experiment.identity import build_experiment_identity, build_experiment_manifest
from medseg.utils.serialization import write_json


@dataclass(frozen=True)
class RunDirectories:
    """Standard directory bundle for one experiment run."""

    root: Path
    checkpoints: Path
    logs: Path
    predictions: Path
    figures: Path


def create_run_directories(
    config: AppConfig,
    timestamp: str | None = None,
) -> RunDirectories:
    """Create the canonical run directory tree for one experiment."""

    identity = build_experiment_identity(config)
    run_id = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(config.experiment.output_root)
        / f"dataset={identity.dataset_name}"
        / f"dataset_version={identity.dataset_version}"
        / f"split_definition={identity.split_definition_name}"
        / f"model={identity.model_name}"
        / f"seed={identity.seed}"
        / f"experiment={identity.experiment_name}"
        / run_id
    )
    checkpoints = root / "checkpoints"
    logs = root / "logs"
    predictions = root / "predictions"
    figures = root / "figures"

    for path in (root, checkpoints, logs, predictions, figures):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = root / config.experiment.manifest_filename
    write_json(manifest_path, build_experiment_manifest(config))

    return RunDirectories(
        root=root,
        checkpoints=checkpoints,
        logs=logs,
        predictions=predictions,
        figures=figures,
    )
