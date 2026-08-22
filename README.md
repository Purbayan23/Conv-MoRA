# Conv-MoRA Medical Image Segmentation

This repository is a research-oriented PyTorch codebase for medical image segmentation. Its baseline is a reference-style U-Net 2D implementation, followed by controlled ConvLoRA adaptation experiments on ISIC skin-lesion segmentation.

The implementation remains configuration-driven and modular so that dataset adapters, model components, and future parameter-efficient adaptation studies can be evaluated without changing the core training and evaluation flow.

## Research Scope

- Independent experiments across medical segmentation datasets
- Reusable dataset, preprocessing, training, validation, inference, metric, and experiment infrastructure
- Dataset-specific parsing, preprocessing, split definitions, and metadata isolated in dataset adapters
- Centralized YAML/dataclass configuration for datasets, models, optimization, runtime, and checkpoints
- Explicit experiment identity and frozen manifests for reproducible adaptation studies
- ConvLoRA adaptation with the source U-Net and ESH held as independently controlled components

## Project Status

### Source U-Net Baseline

- The original/reference U-Net 2D baseline has been implemented and validated.
- Source dataset: ISIC-2016 binary skin-lesion segmentation
- Source data: 900 valid image/mask pairs
- Deterministic split: 720 training samples and 180 validation samples, seed 42
- Input resolution: 256x256
- Model: U-Net 2D with `n_filters_init=16`
- Source training: 50 epochs, batch size 32, learning rate 0.001
- ConvLoRA was disabled during source training.

### Source Baseline Result

The best source checkpoint was selected using validation Dice. An independent ISIC-2016 validation evaluation produced:

| Metric | Value |
| --- | ---: |
| Loss | 0.1923623972 |
| Dice | 0.9038399019 |
| IoU | 0.8245510239 |
| Precision | 0.9248537477 |
| Recall | 0.8837597325 |

These are validation results, not official test performance.

### Early Segmentation Head

- ESH training occurs after source U-Net training.
- The source U-Net is frozen during ESH training.
- ESH consumes level-3 encoder features.
- ESH training uses 20 epochs.
- Best validation Dice: `0.8821184004`
- The selected ESH checkpoint is frozen during target adaptation.

### ConvLoRA Target Adaptation

- Target dataset: ISIC-2017
- Target adaptation manifest: 1003 samples
- Adaptation subset: 803 samples
- Consistency subset: 200 samples
- Adaptation/consistency intersection: 0
- Target evaluation: a separate labeled 251-image evaluation set
- Target masks are not accessed during adaptation.
- ConvLoRA insertion scope: `init_path`, `down1`, `down2`, `down3`
- ConvLoRA rank: 2
- ConvLoRA alpha: 2
- Trainable parameters: 54,870
- Base U-Net parameters remain frozen.
- ESH remains frozen.
- The first adaptation experiment used 5 epochs.
- Best consistency Dice in the first adaptation experiment: `0.8213968026`

### First Target Result

The following is the first ConvLoRA adaptation experiment. It showed an observed negative-transfer result.

| Evaluation | Loss | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source-only ISIC-2017 target evaluation | 0.246095 | 0.825841 | 0.703347 | 0.745641 | 0.925374 |
| ConvLoRA adaptation | 0.297705 | 0.770088 | 0.626132 | 0.651980 | 0.940451 |

Dice change: `-0.055753`.

This result is an observed negative-transfer outcome. It is not evidence of an implementation bug; the adaptation objective and indirect consistency-based checkpoint criterion can improve agreement with the source/ESH signal without improving target ground-truth segmentation.

## Implementation Audit

A read-only implementation audit found the following contracts correct:

- Source checkpoint loading and base-parameter freezing
- ConvLoRA insertion scope
- ConvLoRA `W + BA(alpha/r)` formulation
- ESH freezing
- Target-label isolation during adaptation
- Disjoint adaptation and consistency subsets
- Isolation of target evaluation from adaptation and checkpoint selection

The audit also identified these protocol/design concerns:

- Consistency-based checkpoint selection is indirect because it compares source and ESH predictions rather than target ground truth.
- The initial adaptation path used non-detached pseudo-labels, allowing gradients through the pseudo-label computation.
- AdaBN behavior is present through BatchNorm running-statistic updates, while the affine-parameter configuration field is currently not wired.

## Current Experiment

Experiment 2A is currently being run. Its only intended change from Experiment 1 is:

```yaml
detach_pseudo_labels: true
```

The source checkpoint, ESH checkpoint, manifests, split seed, ConvLoRA rank, alpha, insertion scope, optimizer, learning rate, and epoch settings remain unchanged. Experiment 2A results are not reported here because the experiment is still running.

## Explicitly Out of Scope Right Now

- MoRA implementation
- Fisher computation or selection
- xLSTM, continual learning, or automatic cross-dataset weight transfer
- Additional target datasets beyond the current ISIC-2017 adaptation study

## Current Structure

- `configs/`: YAML configuration groups for datasets, preprocessing, augmentation, models, optimization, runtime, checkpoints, and experiments
- `src/medseg/`: modular PyTorch implementation for datasets, models, losses, metrics, training, validation, inference, and adaptation
- `docs/`: architectural notes and explicit data/model contracts
- `tests/`: contract-focused and Stage 2 regression tests
- `data/`: expected dataset layout documentation
- `outputs/`: run artifacts such as manifests, logs, checkpoints, figures, and predictions

## Getting Started

1. Create an environment with Python 3.10 or newer.
2. Install the project dependencies with `pip install -r requirements/base.txt`.
3. Optionally install development dependencies with `pip install -r requirements/dev.txt`.
4. Review [docs/contracts.md](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/docs/contracts.md), [configs/config.yaml](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/configs/config.yaml), and [data/README.md](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/data/README.md).

Do not interpret Experiment 2A as a completed result until its run has finished and its separate target evaluation has been performed.
