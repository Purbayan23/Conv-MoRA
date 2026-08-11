"""Lightweight qualitative visualization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def default_overlay_name(sample_id: str) -> str:
    """Return a deterministic filename for one qualitative overlay."""

    return f"{sample_id}_overlay.png"


def save_qualitative_examples(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    sample_ids: Sequence[str] | None,
    output_dir: Path,
    threshold: float = 0.5,
    start_index: int = 0,
    max_examples: int = 5,
) -> int:
    """Save input, ground truth, and binary prediction panels as PNG files."""

    if max_examples <= start_index:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (torch.sigmoid(logits) >= threshold).to(dtype=torch.float32)
    count = min(images.shape[0], max_examples - start_index)
    for batch_index in range(count):
        sample_id = (
            str(sample_ids[batch_index])
            if sample_ids is not None and batch_index < len(sample_ids)
            else f"example_{start_index + batch_index:03d}"
        )
        image = _image_to_pil(images[batch_index])
        ground_truth = _mask_to_pil(masks[batch_index])
        prediction = _mask_to_pil(predictions[batch_index])
        panel = Image.new("RGB", (image.width * 3, image.height))
        panel.paste(image, (0, 0))
        panel.paste(ground_truth, (image.width, 0))
        panel.paste(prediction, (image.width * 2, 0))
        panel.save(output_dir / default_overlay_name(sample_id))
    return count


def _image_to_pil(image: torch.Tensor) -> Image.Image:
    tensor = image.detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError("Qualitative images must have CHW shape.")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] < 3:
        raise ValueError("Qualitative images must have one or three channels.")
    tensor = tensor[:3]
    if tensor.min() < 0.0 or tensor.max() > 1.0:
        minimum, maximum = tensor.min(), tensor.max()
        tensor = (tensor - minimum) / (maximum - minimum).clamp_min(1.0e-08)
    array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _mask_to_pil(mask: torch.Tensor) -> Image.Image:
    tensor = mask.detach().cpu().float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError("Qualitative masks must have HW or 1HW shape.")
    array = (tensor > 0.5).numpy().astype(np.uint8) * 255
    return Image.fromarray(array, mode="L").convert("RGB")
