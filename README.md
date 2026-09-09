# Conv-MoRA Medical Image Segmentation

This repository investigates parameter-efficient domain adaptation for convolutional U-Net models in medical image segmentation. ConvLoRA is the established adaptation baseline, and ConvMoRA is the planned subsequent research direction. ConvMoRA has not yet been implemented or evaluated.

The reported ConvLoRA experiments use the `UNet2D` architecture from the ConvLoRA reference project. The repository also retains an independent original-style Ronneberger U-Net baseline/reference; it is documented separately from the ConvLoRA `UNet2D` experiments.

## Research Scope

- Independent source-training and target-adaptation experiments across medical segmentation datasets
- Binary skin-lesion segmentation at 256x256 resolution
- PyTorch implementation with YAML/dataclass configuration
- Dataset-specific adapters, preprocessing, manifests, and metadata
- Unlabeled target images for adaptation and a separate frozen labeled target-evaluation split
- No target-mask access during adaptation or adaptation checkpoint selection
- No continual-learning or sequential LoRA-to-MoRA training
- No automatic cross-dataset weight transfer

## Paper, Reference, And Project Mapping

At the paper level, ConvLoRA is specified as full-encoder convolutional adaptation with frozen pretrained model parameters apart from adaptation parameters, AdaBN through BatchNorm running-statistic adaptation, and ESH/self-training with pseudo-labels. The original experiments were performed on CC359, a 3D brain MRI skull-stripping dataset. The present ISIC experiments do not reproduce that CC359 benchmark.

The authors' public executable repository contains an implementation inconsistency. In `adaptation.py`, `target_adaptation()` constructs the base `UNet2D` and freezes its parameters before entering the adaptation-method branches. The `constrained_lora_down3` training branch then only enables parameters whose names contain `"bn"` in `init_path`, `down1`, `down2`, and `down3`; it does not invoke `replace_layers(...)` or `mark_only_lora_as_trainable(...)`. The optimizer is created afterward over this unchanged model. In addition, checks of the form `isinstance(name, nn.BatchNorm2d)` operate on parameter-name strings and are therefore always false. The branch consequently trains selected BatchNorm-named parameters in a standard U-Net rather than executing ConvLoRA insertion, and should not be treated as a reliable executable implementation of the paper-level full-encoder ConvLoRA configuration. The `test.py` branch corresponding to `lora:down3` explicitly inserts LoRA into `init_path`, `down1`, `down2`, and `down3`, while leaving the decoder/output path without LoRA. This test/inference-path evidence supports the intended four-stage placement, but it does not prove that the corresponding training branch was correctly implemented.

The present project follows the paper-level architectural specification and the executable evidence of the intended encoder adapter placement rather than claiming byte-for-byte reproduction of the reference repository. The implemented four-stage placement is `init_path`, `down1`, `down2`, and `down3`; the decoder and output path do not contain ConvLoRA adapters. The subsequent controlled ablation identified 3x3-only ConvLoRA with frozen BN running statistics as the preferred baseline for the forthcoming ConvLoRA-versus-ConvMoRA comparison.

## Dataset And Split Protocol

ISIC-2016 is the source domain and ISIC-2017 is the target domain.

| Dataset or split | Samples | Protocol |
| --- | ---: | --- |
| ISIC-2016 usable image-mask pairs | 900 | Source dataset |
| ISIC-2016 training split | 720 | Split seed 42 |
| ISIC-2016 validation split | 180 | Split seed 42 |
| ISIC-2017 target adaptation manifest | 1003 | Deterministic split seed 42 |
| ISIC-2017 adaptation subset | 803 | Unlabeled during adaptation |
| ISIC-2017 consistency subset | 200 | Disjoint from adaptation; used for adaptation checkpoint selection |
| ISIC-2017 target evaluation | 251 | Separate labeled evaluation set |

The adaptation and consistency subsets are disjoint. The target-evaluation set is separate from both subsets. Target masks are not opened during adaptation or consistency-based checkpoint selection.

Historical ConvLoRA adaptation used `drop_last=True`, so 800 of the 803 adaptation samples were processed per epoch. The BN-only ablation used `drop_last=False` and processed all 803 adaptation samples per pass. The two behaviors must not be conflated.

## Source Model And Training

The source model is the ConvLoRA project's `UNet2D`, not the independent original Ronneberger U-Net baseline. Source training used ISIC-2016 with the following protocol:

- 50 epochs
- Batch size 32
- Adam optimizer
- Learning rate 0.001
- BCE reference loss
- No scheduler
- Seed 42
- Input resolution 256x256
- `n_filters_init=16`
- ConvLoRA disabled
- Best checkpoint selected by source validation Dice

### Source Validation Result

The independent ISIC-2016 validation evaluation was:

| Metric | Value |
| --- | ---: |
| Loss | 0.1923623972 |
| Dice | 0.9038399019 |
| IoU | 0.8245510239 |
| Precision | 0.9248537477 |
| Recall | 0.8837597325 |

This is a source validation result, not an official ISIC test-set result.

## Early Segmentation Head

The Early Segmentation Head (ESH) was trained independently using the frozen source U-Net. It consumed level-3 features with 128 input channels and used three convolutional stages with BatchNorm/ReLU followed by 8x bilinear upsampling.

- Epochs: 20
- Optimizer: Adam
- Best validation Dice: 0.8821184004
- Independent evaluation Dice: 0.8821184004
- ESH state: frozen during target adaptation

Target masks were not used during target adaptation.

## Historical ConvLoRA Baseline

The historical ConvLoRA adaptation used:

- Encoder insertion scope: `[init_path, down1, down2, down3]`
- Rank: `r=2`
- Alpha: `2`
- Decoder and output: no ConvLoRA adapters
- Adapted convolutional layers: 28
- Total model parameters: 2,489,928
- Trainable ConvLoRA parameters: 54,870

| Scope | Trainable ConvLoRA parameters |
| --- | ---: |
| `init_path` | 3,798 |
| `down1` | 7,296 |
| `down2` | 14,592 |
| `down3` | 29,184 |

The historical scope includes the first RGB-to-feature input convolution, as implemented by this repository. Historical results must not be rewritten to claim that this convolution was excluded.

## Completed Target Adaptation Experiments

All target adaptations used the ISIC-2017 manifests, the 803/200 adaptation-consistency split, a separate 251-image target evaluation set, and unlabeled target images during adaptation. Unless explicitly noted, historical ConvLoRA runs processed 800 of the 803 adaptation samples per epoch because of `drop_last=True`.

### Source-Only Target Evaluation

The source-only model established the target-domain reference:

| Metric | Value |
| --- | ---: |
| Loss | 0.2461 |
| Dice | 0.8258 |
| IoU | 0.7033 |
| Precision | 0.7456 |
| Recall | 0.9254 |

### BN-Only / AdaBN Ablation

This diagnostic ablation tested whether BatchNorm running-statistic adaptation alone explained the ConvLoRA result. All model parameters were frozen; ConvLoRA, the optimizer, ESH, pseudo-label training, and pseudo-label loss were absent. Only the 803 target adaptation images were used to collect BatchNorm statistics. Target masks and the 200-image consistency subset were not used. Five passes were performed with `drop_last=False`, processing all 803 images per pass.

| Pass | Loss | Dice | IoU | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2461 | 0.8258 | 0.7033 | 0.7456 | 0.9254 |
| 1 | 0.2217 | 0.8308 | 0.7105 | 0.7676 | 0.9052 |
| 2 | 0.2197 | 0.8312 | 0.7112 | 0.7711 | 0.9015 |
| 3 | 0.2196 | 0.8313 | 0.7112 | 0.7713 | 0.9013 |
| 4 | 0.2196 | 0.8313 | 0.7112 | 0.7714 | 0.9013 |
| 5 | 0.2196 | 0.8313 | 0.7113 | 0.7714 | 0.9012 |

BN-only adaptation produced a modest improvement over source-only evaluation. This observation does not establish that AdaBN is universally beneficial.

### ConvLoRA + AdaBN

The first ConvLoRA adaptation used rank 2, alpha 2, insertion scope `[init_path, down1, down2, down3]`, a frozen ESH, unlabeled target images, consistency-based checkpoint selection, 5 epochs, learning rate 0.0001, and batch size 32. Historical `drop_last=True` behavior processed 800 of 803 adaptation samples per epoch. Target masks were not used during adaptation.

| Epoch | Adaptation loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5354 | 0.8117 |
| 2 | 0.5242 | 0.8102 |
| 3 | 0.5088 | 0.8107 |
| 4 | 0.4934 | 0.8202 |
| 5 | 0.4793 | 0.8214 |

The best consistency Dice was `0.8213968026` at epoch 5. Independent target evaluation produced:

| Metric | Value |
| --- | ---: |
| Loss | 0.297705 |
| Dice | 0.770088 |
| IoU | 0.626132 |
| Precision | 0.651980 |
| Recall | 0.940451 |

Dice decreased by approximately `0.055753` from the source-only reference. This is negative transfer under the stated protocol.

### ConvLoRA + AdaBN With Detached Pseudo-Labels

Experiment 2A used the same protocol as the first ConvLoRA experiment, with `detach_pseudo_labels=true` as the only experimental change.

| Epoch | Adaptation loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5358 | 0.8114 |
| 2 | 0.5256 | 0.8079 |
| 3 | 0.5108 | 0.8064 |
| 4 | 0.4958 | 0.8163 |
| 5 | 0.4821 | 0.8168 |

The best consistency Dice was `0.8168` at epoch 5. Target evaluation produced:

| Metric | Value |
| --- | ---: |
| Loss | 0.297551 |
| Dice | 0.768895 |
| IoU | 0.624557 |
| Precision | 0.654935 |
| Recall | 0.930868 |

Dice changed by approximately `-0.001193` relative to the non-detached experiment. Under this configuration, pseudo-label detachment did not materially change the outcome. The result does not rule out all pseudo-label-related effects.

### ConvLoRA + Frozen BatchNorm

This ablation tested whether the negative interaction arose from updating BatchNorm running statistics jointly with ConvLoRA. ConvLoRA used rank 2, alpha 2, scope `[init_path, down1, down2, down3]`, and a frozen ESH. BN running statistics and all non-adapter parameters were frozen. The 803/200 split and 5-epoch consistency-selection protocol were preserved; target masks were not used during adaptation.

| Epoch | Adaptation loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5565 | 0.8160 |
| 2 | 0.5271 | 0.8043 |
| 3 | 0.4899 | 0.8151 |
| 4 | 0.4481 | 0.8388 |
| 5 | 0.4118 | 0.8572 |

The best consistency Dice was `0.8572` at epoch 5. Target evaluation produced:

| Metric | Value |
| --- | ---: |
| Loss | 0.1975 |
| Dice | 0.8355 |
| IoU | 0.7175 |
| Precision | 0.9068 |
| Recall | 0.7747 |

Under the present experimental protocol, allowing BN running statistics to adapt jointly with ConvLoRA substantially degraded target-domain performance, whereas freezing BN running statistics removed this degradation. This pattern strongly implicates the interaction between ConvLoRA updates and BN running-statistic adaptation under this protocol. It does not establish that AdaBN is universally harmful.

### ConvLoRA 3x3-Only + Frozen BatchNorm

This ablation tested whether 2x2 strided downsampling convolutions were necessary. ConvLoRA was applied only to 3x3 convolutions within `[init_path, down1, down2, down3]`; the 2x2 stride-2 downsampling convolutions and decoder/output remained unchanged. Rank was 2, alpha was 2, BN running statistics were frozen, and ESH/self-training, the data split, optimizer, learning rate, and epoch count were unchanged. Target masks were not used during adaptation.

The 3x3-only model had 52,182 trainable ConvLoRA parameters, compared with 54,870 for the historical full-encoder variant. This is a reduction of 2,688 parameters, approximately 4.9%.

| Epoch | Adaptation loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5566 | 0.8146 |
| 2 | 0.5279 | 0.8021 |
| 3 | 0.4914 | 0.8192 |
| 4 | 0.4498 | 0.8350 |
| 5 | 0.4138 | 0.8552 |

The best consistency Dice was `0.8552` at epoch 5. Independent target evaluation produced:

| Metric | Value |
| --- | ---: |
| Loss | 0.1963 |
| Dice | 0.8395 |
| IoU | 0.7233 |
| Precision | 0.9053 |
| Recall | 0.7825 |

Relative to full-encoder frozen-BN ConvLoRA, target Dice changed from 0.8355 to 0.8395, an improvement of 0.0040. IoU changed from 0.7175 to 0.7233, an improvement of 0.0058. Consistency Dice was slightly lower, changing from 0.8572 to 0.8552. The result provides evidence that 2x2 downsampling adapters were not necessary to obtain the observed target-domain improvement in this protocol; it does not establish universal superiority.

## Implementation Audit

Verified:

- Source checkpoint loading and freezing
- ConvLoRA insertion scope
- ConvLoRA `W + BA(alpha/r)` formulation
- Historical trainable parameter count of 54,870
- ESH freezing during target adaptation
- No target-mask access during adaptation
- Disjoint adaptation and consistency subsets
- Separation of target evaluation from adaptation

The following protocol and design concerns remain documented:

- Consistency-based checkpoint selection is indirect because it does not use target-label Dice.
- The first ConvLoRA experiment used non-detached pseudo-labels; Experiment 2A tested detachment.
- The `adabn_train_affine` configuration field is not wired to runtime trainability. The reported adaptation experiments used frozen BN affine parameters.

The paper-level method, the authors' executable repository, and the present project are therefore distinct scientific references: full-encoder ConvLoRA plus AdaBN plus ESH/self-training on CC359; an executable repository with an inconsistent/incomplete `constrained_lora_down3` training branch and test-path evidence for four-stage placement; and an ISIC implementation that follows the intended four-stage architecture while explicitly recording dataset, metric, protocol, and BN-ablation departures.

These observations distinguish protocol/design limitations from implementation defects. The observed negative transfer is not presented as evidence of an implementation bug.

## Metrics And Evaluation Semantics

The original ConvLoRA paper used Surface Dice Score for its CC359 evaluation. CC359 is a 3D brain MRI skull-stripping dataset in which accurate anatomical boundary delineation is important, making a surface-based metric appropriate for that evaluation setting. NIfTI and nibabel are data representation and loading choices; they are not the reason Surface Dice was selected.

The present project evaluates 2D dermoscopic ISIC skin-lesion segmentation. Binary Dice is the primary segmentation metric, with IoU, precision, and recall reported as complementary metrics. Metric selection is task-dependent; Surface Dice is not characterized as unsuitable for 2D segmentation.

Final target-evaluation metrics use the labeled 251-image ISIC evaluation set. Adaptation checkpoint selection uses aggregated binary Dice on the unlabeled 200-image consistency subset and does not use target ground-truth labels. Project binary Dice values are therefore not numerically equivalent to the Surface Dice scores reported for CC359. The metric difference does not invalidate the internal ConvLoRA-versus-ConvMoRA comparison because both methods use the same evaluation protocol.

## Current ConvLoRA Reference For ConvMoRA Comparison

The current preferred ConvLoRA reference for the subsequent ConvMoRA comparison is the 3x3-only frozen-BN variant:

- 3x3-only ConvLoRA within `[init_path, down1, down2, down3]`
- Rank 2
- Alpha 2
- Frozen BN running statistics
- ESH/self-training unchanged
- Target-evaluation Dice: 0.8395
- Trainable adapter parameters: 52,182

## Planned ConvMoRA Design Constraint

ConvMoRA is the next research stage, but implementation has not begun. The planned comparison will use the grouped row/column-sharing formulation with a square matrix under parameter matching.

The ConvMoRA comparison should change only the adapter parameterization while keeping the encoder insertion scope, 3x3-only adapter restriction, BN treatment, ESH, pseudo-label objective, target adaptation data, adaptation/consistency split, optimizer, training schedule, and evaluation protocol fixed wherever applicable.

For each convolution, parameter matching must be calculated from the flattened convolution dimensions. The reference first RGB-to-64 convolution requires special treatment because its flattened input dimension is only 9. If an equal-budget square MoRA dimension selected from parameter matching exceeds the smaller flattened dimension, a genuine high-rank/compressive MoRA mapping cannot be claimed. Exact parameter calculations and layer eligibility must therefore be established before implementation.

## Explicitly Out Of Scope

- ConvMoRA/MoRA implementation or evaluation
- Continual learning
- Automatic cross-dataset weight transfer
- Additional dataset experiments

## Current Structure

- `configs/`: YAML configuration groups for datasets, preprocessing, augmentation, models, optimization, runtime, checkpoints, and experiments
- `src/medseg/`: modular PyTorch implementation for datasets, models, losses, metrics, training, validation, inference, and adaptation
- `docs/`: architecture, contract, and reference-mapping notes
- `tests/`: contract-focused and Stage 2 regression tests
- `data/`: expected dataset-layout documentation
- `outputs/`: run artifacts such as manifests, logs, checkpoints, figures, and predictions

## Getting Started

1. A Python 3.10 or newer environment is required.
2. Project dependencies are installed with `pip install -r requirements/base.txt`.
3. Development dependencies can be installed with `pip install -r requirements/dev.txt`.
4. Project contracts and configuration are documented in [docs/contracts.md](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/docs/contracts.md), [configs/config.yaml](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/configs/config.yaml), and [data/README.md](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/data/README.md).
