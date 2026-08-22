# Conv-MoRA Medical Image Segmentation

This repository contains a configuration-driven PyTorch codebase for medical image segmentation. The source U-Net baseline was implemented using the project's defined source-training configuration. The model is based on the classical U-Net architecture introduced by Ronneberger et al. (2015). The implementation is not described as an exact reproduction of the complete architecture and training procedure from the original paper.

The configuration names `unet2d_source`, `bce_reference`, and `adam_reference` identify project-defined experimental/reference configurations. These names do not establish exact reproduction of the original paper's complete architecture or training procedure.

## Research Scope

- Independent experiments across medical segmentation datasets
- Dataset, preprocessing, training, validation, inference, metric, and experiment infrastructure
- Dataset-specific parsing, preprocessing, split definitions, and metadata isolated in dataset adapters
- Centralized YAML/dataclass configuration for datasets, models, optimization, runtime, and checkpoints
- Explicit experiment identity and frozen manifests for reproducible adaptation studies
- ConvLoRA adaptation with the source U-Net and Early Segmentation Head treated as separate experimental components

## Experimental History

### Source U-Net Baseline

ISIC-2016 was used as the source dataset. The dataset adapter identified 900 valid image/mask pairs and generated a deterministic split containing 720 training samples and 180 validation samples with split seed 42. Images were processed at 256x256 resolution. The U-Net used `n_filters_init=16`, and ConvLoRA was disabled during source training. Source training was conducted for 50 epochs with batch size 32 and learning rate 0.001.

The best source checkpoint was selected using validation Dice. An independent ISIC-2016 validation evaluation produced the following result:

| Metric | Value |
| --- | ---: |
| Loss | 0.1923623972 |
| Dice | 0.9038399019 |
| IoU | 0.8245510239 |
| Precision | 0.9248537477 |
| Recall | 0.8837597325 |

This is a validation result and is not an official test-set result.

### Early Segmentation Head

The source U-Net was frozen before Early Segmentation Head training. The ESH was trained using source training and validation data, consumed level-3 encoder features, and was trained for 20 epochs. The best validation Dice was `0.8821184004`. The selected ESH checkpoint was frozen during target adaptation.

### First ConvLoRA Adaptation Experiment

The target dataset was ISIC-2017. The target adaptation manifest contained 1003 samples. A deterministic split assigned 803 samples to adaptation and 200 samples to the consistency subset; the two subsets had zero intersection. A separate labeled 251-image target-evaluation set was used after adaptation. Target masks were not accessed during adaptation.

ConvLoRA was inserted into `init_path`, `down1`, `down2`, and `down3` with rank 2 and alpha 2. The number of trainable parameters was 54,870. The source U-Net parameters and ESH parameters remained frozen. The first adaptation experiment used 5 epochs, and its best consistency Dice was `0.8213968026`.

The first target evaluation produced the following comparison:

| Evaluation | Loss | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source-only ISIC-2017 target evaluation | 0.246095 | 0.825841 | 0.703347 | 0.745641 | 0.925374 |
| First ConvLoRA adaptation | 0.297705 | 0.770088 | 0.626132 | 0.651980 | 0.940451 |

The Dice change was `-0.055753`. Target performance decreased after adaptation. This result does not constitute evidence of an implementation failure because the implementation audit did not identify a structural ConvLoRA or target-leakage error.

## Implementation Audit

The read-only implementation audit verified the following:

- Source checkpoint loading and freezing
- ConvLoRA insertion scope
- ConvLoRA `W + BA(alpha/r)` formulation
- A trainable parameter count of 54,870
- ESH freezing
- No access to target masks during adaptation
- Disjoint adaptation and consistency subsets
- Separation of target evaluation from adaptation

The following implementation observation was recorded:

The configuration field `adabn_train_affine` was found to have no effect on runtime trainability because trainable parameters were controlled by `mark_only_adapter_as_trainable()`. The current experiment used frozen BatchNorm affine parameters.

BatchNorm running-statistic updates were present during target adaptation.

The adaptation procedure selected the checkpoint using an unlabeled source-to-ESH consistency measure rather than target-label Dice. This differs from the label-aware selection path found in the executable reference adaptation implementation. This is a protocol difference, not an implementation bug.

## Experiment 2A: Detached Pseudo-Labels

A configuration-controlled option named `detach_pseudo_labels` was added. Its default behavior remains `false`. Experiment 2A used `detach_pseudo_labels=true`. No other experimental setting was changed. The source checkpoint, ESH checkpoint, target manifests, split seed, ConvLoRA rank, alpha, insertion scope, optimizer, learning rate, and epoch count remained unchanged.

The adaptation trajectory was:

| Epoch | Adaptation loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5358 | 0.8114 |
| 2 | 0.5256 | 0.8079 |
| 3 | 0.5108 | 0.8064 |
| 4 | 0.4958 | 0.8163 |
| 5 | 0.4821 | 0.8168 |

The best consistency Dice was `0.8168` at epoch 5.

Experiment 2A target evaluation produced:

| Metric | Value |
| --- | ---: |
| Loss | 0.297551 |
| Dice | 0.768895 |
| IoU | 0.624557 |
| Precision | 0.654935 |
| Recall | 0.930868 |

The Dice change relative to Experiment 1 was `-0.001193`. The Dice change relative to the source-only result was `-0.056946`.

Detaching the pseudo-labels did not improve target Dice under the tested configuration. Target Dice changed from `0.770088` to `0.768895`. This conclusion is limited to Experiment 2A and does not establish that pseudo-label detachment is generally ineffective.

## Current Status

The first two adaptation experiments have been completed and archived. The next planned investigation is isolation of BatchNorm/AdaBN behavior. That investigation has not been completed.

## Explicitly Out of Scope Right Now

- MoRA implementation
- Fisher computation or selection
- xLSTM, continual learning, or automatic cross-dataset weight transfer
- Results from future adaptation or dataset experiments

## Current Structure

- `configs/`: YAML configuration groups for datasets, preprocessing, augmentation, models, optimization, runtime, checkpoints, and experiments
- `src/medseg/`: modular PyTorch implementation for datasets, models, losses, metrics, training, validation, inference, and adaptation
- `docs/`: architectural notes and explicit data/model contracts
- `tests/`: contract-focused and Stage 2 regression tests
- `data/`: expected dataset layout documentation
- `outputs/`: run artifacts such as manifests, logs, checkpoints, figures, and predictions

## Getting Started

1. A Python 3.10 or newer environment is required.
2. Project dependencies are installed with `pip install -r requirements/base.txt`.
3. Development dependencies can be installed with `pip install -r requirements/dev.txt`.
4. The project contracts and configuration are documented in [docs/contracts.md](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/docs/contracts.md), [configs/config.yaml](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/configs/config.yaml), and [data/README.md](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/data/README.md).
