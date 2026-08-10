"""Minimal model-agnostic training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from medseg.config.schema import AppConfig
from medseg.experiment.identity import build_experiment_identity
from medseg.models.builder import describe_model_request
from medseg.optimization.builder import build_optimizer_spec, build_scheduler_spec
from medseg.training.checkpointing import (
    build_checkpoint_paths,
    load_checkpoint,
    save_checkpoint,
)
from medseg.utils.serialization import write_json
from medseg.validation.evaluator import Evaluator


@dataclass
class Trainer:
    """Central coordinator for the baseline train/validation experiment."""

    config: AppConfig
    component_summary: dict[str, str]
    model: nn.Module | None = None
    optimizer: Optimizer | None = None
    loss_fn: nn.Module | None = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    device: torch.device | str = "cpu"

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        loss_fn: nn.Module | None = None,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        device: torch.device | str = "cpu",
    ) -> "Trainer":
        """Create a trainer while keeping component construction external."""

        identity = build_experiment_identity(config)
        optimizer_spec = build_optimizer_spec(config)
        scheduler_spec = build_scheduler_spec(config)
        summary = {
            "dataset": identity.dataset_name,
            "dataset_version": identity.dataset_version,
            "split_definition": identity.split_definition_name,
            "model": describe_model_request(config),
            "preprocessing": config.preprocessing.name,
            "augmentation": config.augmentation.name,
            "loss": config.loss.name,
            "metrics": ", ".join(config.metrics.names),
            "optimizer": optimizer_spec.name,
            "scheduler": scheduler_spec.name,
            "checkpoint_load_enabled": str(config.checkpoint.load_enabled).lower(),
        }
        return cls(
            config=config,
            component_summary=summary,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

    def fit(
        self,
        epochs: int | None = None,
        run_directory: Path | None = None,
        resume_path: Path | None = None,
    ) -> list[dict[str, float | int]]:
        """Train the model and save the best validation-Dice checkpoint."""

        if any(
            component is None
            for component in (self.model, self.optimizer, self.loss_fn, self.train_loader, self.val_loader)
        ):
            raise ValueError(
                "Trainer.fit requires model, optimizer, loss_fn, train_loader, and val_loader."
            )

        assert self.model is not None
        assert self.optimizer is not None
        assert self.loss_fn is not None
        assert self.train_loader is not None
        assert self.val_loader is not None

        total_epochs = self.config.experiment.max_epochs if epochs is None else epochs
        checkpoint_paths = build_checkpoint_paths(run_directory) if run_directory is not None else None
        evaluator = Evaluator.from_config(self.config)
        history: list[dict[str, float | int]] = []
        start_epoch = 1
        best_val_dice = float("-inf")

        if resume_path is not None:
            checkpoint = load_checkpoint(
                resume_path,
                self.model,
                optimizer=self.optimizer,
                device=self.device,
            )
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_dice = float(checkpoint.get("best_val_dice", best_val_dice))
            history = list(checkpoint.get("history", []))

        for epoch in range(start_epoch, total_epochs + 1):
            train_loss = self._train_epoch()
            validation = evaluator.evaluate(
                model=self.model,
                dataloader=self.val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
            )
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation["loss"],
                "val_dice": validation["dice"],
                "val_iou": validation["iou"],
            }
            history.append(record)
            print(
                f"Epoch {epoch:03d}/{total_epochs:03d} | "
                f"train_loss={train_loss:.4f} | val_loss={validation['loss']:.4f} | "
                f"val_dice={validation['dice']:.4f} | val_iou={validation['iou']:.4f}"
            )

            if checkpoint_paths is not None:
                save_checkpoint(
                    checkpoint_paths.latest,
                    self.model,
                    self.optimizer,
                    epoch,
                    max(best_val_dice, validation["dice"]),
                    history,
                )
                if validation["dice"] > best_val_dice:
                    best_val_dice = validation["dice"]
                    save_checkpoint(
                        checkpoint_paths.best,
                        self.model,
                        self.optimizer,
                        epoch,
                        best_val_dice,
                        history,
                    )
            else:
                best_val_dice = max(best_val_dice, validation["dice"])

            if run_directory is not None:
                write_json(
                    run_directory / "history.json",
                    {"best_val_dice": best_val_dice, "history": history},
                )

        return history

    def _train_epoch(self) -> float:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.loss_fn is not None
        assert self.train_loader is not None

        self.model.train()
        total_loss = 0.0
        sample_count = 0
        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite.")
            loss.backward()
            self.optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("Cannot train on an empty dataloader.")
        return total_loss / sample_count
