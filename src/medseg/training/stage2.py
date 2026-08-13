"""Small, explicit helpers for the Stage 2 ConvLoRA adaptation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from medseg.config.schema import AppConfig
from medseg.losses.base import BaseLoss
from medseg.metrics.binary_segmentation import binary_confusion_counts
from medseg.models.architectures.unet2d import UNet2D
from medseg.models.extensions.convlora import (
    apply_convlora,
    freeze_base_model,
    mark_only_adapter_as_trainable,
)
from medseg.models.heads.early_segmentation import EarlySegmentationHead


def load_model_state(
    checkpoint_path: str | Path,
    model: nn.Module,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load either a project checkpoint wrapper or a raw PyTorch state dict."""

    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=strict)
    return dict(payload) if isinstance(payload, Mapping) else {"model_state_dict": payload}


def build_base_unet(config: AppConfig) -> UNet2D:
    """Build the reference UNet2D without adapters for source-side work."""

    model_config = replace(config.model, convlora_enabled=False, encoder_insertion_scope=[])
    return UNet2D.from_config(model_config)


def prepare_adaptation_model(
    config: AppConfig,
    source_checkpoint: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> UNet2D:
    """Load a source model, insert encoder ConvLoRA, and prepare AdaBN mode."""

    model = build_base_unet(config)
    checkpoint = source_checkpoint or config.stage2.source_checkpoint
    if checkpoint:
        load_model_state(checkpoint, model, device=device)
    apply_convlora(
        model,
        rank=config.stage2.convlora_rank,
        alpha=config.stage2.convlora_alpha,
        scope=config.stage2.insertion_scope,
    )
    mark_only_adapter_as_trainable(model)
    # Base BN affine parameters stay frozen; train mode updates running buffers.
    model.train()
    return model.to(device)


def prepare_frozen_esh(
    config: AppConfig,
    esh_checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> EarlySegmentationHead:
    """Load the ESH and freeze it while retaining gradients to its inputs."""

    level = config.stage2.esh_level
    esh = EarlySegmentationHead(
        in_channels=config.model.n_filters_init * (2**level),
        out_channels=config.model.out_channels,
        level=level,
    )
    load_model_state(esh_checkpoint, esh, device=device)
    for parameter in esh.parameters():
        parameter.requires_grad = False
    esh.eval()
    return esh.to(device)


def target_adaptation_step(
    model: UNet2D,
    esh: EarlySegmentationHead,
    images: torch.Tensor,
    loss_fn: BaseLoss | nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the reference-style source-pseudo-label to ESH computation graph."""

    source_logits = model(images)["logits"]
    pseudo_labels = torch.sigmoid(source_logits)
    features = model.encode(images)[_feature_name(esh.level)]
    adapted_logits = esh(features)
    if adapted_logits.shape[-2:] != source_logits.shape[-2:]:
        adapted_logits = F.interpolate(
            adapted_logits,
            size=source_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    loss = loss_fn({"logits": adapted_logits}, pseudo_labels)
    return loss, pseudo_labels, source_logits, adapted_logits


def split_adaptation_dataset(
    dataset: Dataset[Any],
    consistency_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset[Any], Subset[Any], tuple[int, ...], tuple[int, ...]]:
    """Create deterministic, disjoint image-only adaptation and consistency subsets."""

    if not 0.0 < consistency_fraction < 1.0:
        raise ValueError("consistency_fraction must be between 0 and 1.")
    sample_count = len(dataset)
    if sample_count < 2:
        raise ValueError("At least two target samples are required for an internal split.")
    consistency_count = max(1, min(sample_count - 1, int(sample_count * consistency_fraction)))
    permutation = torch.randperm(sample_count, generator=torch.Generator().manual_seed(seed))
    consistency_indices = tuple(sorted(permutation[:consistency_count].tolist()))
    consistency_set = set(consistency_indices)
    adaptation_indices = tuple(
        index for index in range(sample_count) if index not in consistency_set
    )
    return (
        Subset(dataset, list(adaptation_indices)),
        Subset(dataset, list(consistency_indices)),
        adaptation_indices,
        consistency_indices,
    )


def global_binary_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """Compute global binary Dice from aggregated TP, FP, and FN counts."""

    true_positive, false_positive, false_negative = binary_confusion_counts(
        {"logits": logits},
        targets,
        threshold=threshold,
        from_logits=True,
    )
    union = true_positive + false_positive + false_negative
    return (2.0 * true_positive + smooth) / (true_positive + union + smooth)


def train_esh(
    config: AppConfig,
    source_checkpoint: str | Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: BaseLoss | nn.Module,
    device: torch.device | str,
    output_dir: str | Path,
) -> Path:
    """Train the small ESH using labeled source images and masks."""

    source_model = build_base_unet(config).to(device)
    load_model_state(source_checkpoint, source_model, device=device)
    freeze_base_model(source_model)
    source_model.eval()
    esh = EarlySegmentationHead(
        in_channels=config.model.n_filters_init * (2**config.stage2.esh_level),
        out_channels=config.model.out_channels,
        level=config.stage2.esh_level,
    ).to(device)
    optimizer = torch.optim.Adam(esh.parameters(), lr=config.stage2.esh_lr, weight_decay=0.0)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    best_path = output_root / "best.pt"
    latest_path = output_root / "latest.pt"
    best_dice = float("-inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.stage2.esh_epochs + 1):
        esh.train()
        total_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with torch.no_grad():
                features = source_model.encode(images)[_feature_name(esh.level)]
            logits = esh(features)
            loss = loss_fn({"logits": logits}, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = images.shape[0]
            total_loss += float(loss.item()) * count
            sample_count += count
        if sample_count == 0:
            raise ValueError("Cannot train ESH on an empty dataloader.")
        validation = _evaluate_esh(source_model, esh, val_loader, loss_fn, device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / sample_count,
            "val_loss": validation["loss"],
            "val_dice": validation["dice"],
        }
        history.append(record)
        is_best = validation["dice"] > best_dice
        if is_best:
            best_dice = validation["dice"]
        payload = {
            "epoch": epoch,
            "model_state_dict": esh.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_dice,
            "history": history,
            "source_checkpoint": str(source_checkpoint),
            "esh_level": config.stage2.esh_level,
        }
        torch.save(payload, latest_path)
        if is_best:
            torch.save(payload, best_path)
        print(
            f"ESH {epoch:03d}/{config.stage2.esh_epochs:03d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | val_dice={record['val_dice']:.4f}"
        )
    return best_path


def adapt_model(
    config: AppConfig,
    model: UNet2D,
    esh: EarlySegmentationHead,
    dataloader: DataLoader,
    loss_fn: BaseLoss | nn.Module,
    device: torch.device | str,
    output_dir: str | Path,
    consistency_dataloader: DataLoader,
) -> Path:
    """Adapt ConvLoRA parameters and select checkpoints after post-update scoring."""

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable ConvLoRA parameters were found for adaptation.")
    optimizer = torch.optim.Adam(trainable, lr=config.stage2.adaptation_lr, weight_decay=0.0)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    best_path = output_root / "best.pt"
    latest_path = output_root / "latest.pt"
    best_metric = float("-inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.stage2.adaptation_epochs + 1):
        model.train()
        esh.eval()
        total_loss = 0.0
        sample_count = 0
        for batch in dataloader:
            images = batch["image"].to(device)
            loss, _, _, _ = target_adaptation_step(model, esh, images, loss_fn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = images.shape[0]
            total_loss += float(loss.item()) * count
            sample_count += count
        if sample_count == 0:
            raise ValueError("Cannot adapt on an empty dataloader.")
        # Score the current post-update model on a deterministic unlabeled holdout.
        consistency_dice = evaluate_consistency(
            model=model,
            esh=esh,
            dataloader=consistency_dataloader,
            device=device,
        )
        record = {
            "epoch": epoch,
            "adaptation_loss": total_loss / sample_count,
            "consistency_dice_global": consistency_dice,
        }
        history.append(record)
        is_best = consistency_dice > best_metric
        if is_best:
            best_metric = consistency_dice
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_consistency_dice": best_metric,
            "history": history,
            "protocol_name": config.stage2.protocol_name,
            "source_checkpoint": config.stage2.source_checkpoint,
            "esh_checkpoint": config.stage2.esh_checkpoint,
            "target_adaptation_manifest": config.stage2.target_adaptation_manifest,
            "target_eval_manifest": config.stage2.target_eval_manifest,
            "target_labels_allowed_for_training": False,
            "insertion_scope": list(config.stage2.insertion_scope),
            "convlora_rank": config.stage2.convlora_rank,
            "convlora_alpha": config.stage2.convlora_alpha,
            "adabn_train_affine": config.stage2.adabn_train_affine,
            "checkpoint_metric": config.stage2.consistency_metric,
            "consistency_subset_size": len(consistency_dataloader.dataset),
            "adaptation_subset_size": len(dataloader.dataset),
            "consistency_split_seed": config.stage2.consistency_split_seed,
        }
        torch.save(payload, latest_path)
        if is_best:
            torch.save(payload, best_path)
        print(
            f"Adapt {epoch:03d}/{config.stage2.adaptation_epochs:03d} | "
            f"loss={record['adaptation_loss']:.4f} | consistency_dice={consistency_dice:.4f}"
        )
    return best_path


def evaluate_consistency(
    model: UNet2D,
    esh: EarlySegmentationHead,
    dataloader: DataLoader,
    device: torch.device | str,
) -> float:
    """Score source-vs-ESH consistency without updating BN running statistics."""

    model_was_training = model.training
    esh_was_training = esh.training
    model.eval()
    esh.eval()
    true_positive = false_positive = false_negative = 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                source_logits = model(images)["logits"]
                pseudo_labels = torch.sigmoid(source_logits)
                features = model.encode(images)[_feature_name(esh.level)]
                adapted_logits = esh(features)
                if adapted_logits.shape[-2:] != source_logits.shape[-2:]:
                    adapted_logits = F.interpolate(
                        adapted_logits,
                        size=source_logits.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                batch_tp, batch_fp, batch_fn = binary_confusion_counts(
                    {"logits": adapted_logits},
                    pseudo_labels,
                )
                true_positive += batch_tp
                false_positive += batch_fp
                false_negative += batch_fn
    finally:
        model.train(model_was_training)
        esh.train(esh_was_training)
    if true_positive + false_positive + false_negative == 0:
        return 1.0
    return _dice_from_counts(true_positive, false_positive, false_negative)


def _evaluate_esh(
    source_model: UNet2D,
    esh: EarlySegmentationHead,
    dataloader: DataLoader,
    loss_fn: BaseLoss | nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    source_model.eval()
    esh.eval()
    total_loss = 0.0
    sample_count = 0
    true_positive = false_positive = false_negative = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            features = source_model.encode(images)[_feature_name(esh.level)]
            logits = esh(features)
            loss = loss_fn({"logits": logits}, masks)
            batch_tp, batch_fp, batch_fn = binary_confusion_counts({"logits": logits}, masks)
            true_positive += batch_tp
            false_positive += batch_fp
            false_negative += batch_fn
            count = images.shape[0]
            total_loss += float(loss.item()) * count
            sample_count += count
    if sample_count == 0:
        raise ValueError("Cannot evaluate ESH on an empty dataloader.")
    return {
        "loss": total_loss / sample_count,
        "dice": _dice_from_counts(true_positive, false_positive, false_negative),
    }


def _dice_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    union = true_positive + false_positive + false_negative
    return (2.0 * true_positive + 1.0) / (true_positive + union + 1.0)


def _feature_name(level: int) -> str:
    names = ("init_path", "down1", "down2", "down3")
    if level < 0 or level >= len(names):
        raise ValueError(f"Unsupported encoder feature level: {level}")
    return names[level]


__all__ = [
    "adapt_model",
    "build_base_unet",
    "evaluate_consistency",
    "global_binary_dice",
    "load_model_state",
    "prepare_adaptation_model",
    "prepare_frozen_esh",
    "split_adaptation_dataset",
    "target_adaptation_step",
    "train_esh",
]
