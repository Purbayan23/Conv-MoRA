# Stage 2 Reference Mapping And Experiment Record

## Paper And Executable Reference Mapping

The ConvLoRA paper specifies ConvLoRA adapters throughout the encoder, frozen pretrained model parameters apart from adaptation parameters, AdaBN through adaptation of BatchNorm running statistics, and ESH/self-training with pseudo-labels. Its original experiments were performed on CC359, a 3D brain MRI skull-stripping dataset. The present ISIC experiments do not reproduce that CC359 benchmark.

The authors' executable reference is not fully consistent with the paper-level specification. In `adaptation.py`, `target_adaptation()` constructs the base `UNet2D` and freezes its parameters before entering the adaptation-method branches. The `constrained_lora_down3` training branch then only enables parameters whose names contain `"bn"` in `init_path`, `down1`, `down2`, and `down3`; it does not invoke `replace_layers(...)` or `mark_only_lora_as_trainable(...)`. The optimizer is created afterward over this unchanged model. In addition, checks of the form `isinstance(name, nn.BatchNorm2d)` operate on parameter-name strings and are therefore always false. The branch consequently trains selected BatchNorm-named parameters in a standard U-Net rather than executing ConvLoRA insertion, and should not be treated as a reliable executable implementation of the paper's stated full-encoder ConvLoRA configuration. The separate `full_lora` branch has inconsistent model-object flow and is not used as the mapping here.

The authors' `test.py` branch corresponding to `lora:down3` explicitly inserts LoRA into `init_path`, `down1`, `down2`, and `down3`, while leaving the decoder/output path without LoRA. This test/inference-path evidence supports the interpretation that the intended placement was the four encoder stages, but it does not prove that the corresponding training branch was correctly implemented. This is an implementation inconsistency in the reference code, not a claim that the paper's scientific method is invalid.

The present project follows the paper-level architectural specification and executable evidence of the intended encoder adapter placement rather than claiming byte-for-byte reproduction of the reference repository. ConvLoRA is inserted into all four encoder stages, including the first RGB-to-feature convolution; base convolution and BN affine parameters are frozen, and the model remains in train mode so BN running statistics can adapt in the historical AdaBN condition. The ESH is frozen and kept in evaluation mode. Target masks are loaded only by target-evaluation commands, never by adaptation.

The reference selects checkpoints with surface Dice, but this repository intentionally has no surface-distance dependency. Stage 2 uses aggregated binary consistency Dice for adaptation checkpoint selection and records that substitution in checkpoint metadata. ISIC preprocessing uses the existing RGB 256x256 pipeline rather than the reference CC359 NIfTI-slice preprocessing.

## Fixed Dataset And Split Protocol

ISIC-2016 is the source domain and ISIC-2017 is the target domain. The source dataset contains 900 usable image-mask pairs, split with seed 42 into 720 training and 180 validation samples.

The ISIC-2017 target adaptation manifest contains 1003 samples. Seed 42 produces 803 adaptation samples and 200 consistency samples, with zero intersection. The target-evaluation manifest contains 251 labeled samples and is separate from both adaptation subsets.

Historical ConvLoRA adaptation uses `drop_last=True`, processing 800 of the 803 adaptation samples per epoch. The BN-only diagnostic uses `drop_last=False`, processing all 803 adaptation samples per pass. Target masks are not accessed during adaptation or consistency-based checkpoint selection.

## Historical ConvLoRA Parameterization

The historical insertion scope is `[init_path, down1, down2, down3]`, with rank 2 and alpha 2. Decoder and output convolutions do not receive adapters. The scope adapts 28 convolutional layers, including the first RGB input convolution, and contains 54,870 trainable ConvLoRA parameters within a 2,489,928-parameter model.

| Scope | Trainable ConvLoRA parameters |
| --- | ---: |
| `init_path` | 3,798 |
| `down1` | 7,296 |
| `down2` | 14,592 |
| `down3` | 29,184 |

## Completed Experiment Record

| Experiment | BN behavior | Trainable adapters | Target Dice | Target IoU |
| --- | --- | ---: | ---: | ---: |
| Source-only | Source checkpoint, no adaptation | 0 | 0.8258 | 0.7033 |
| BN-only / AdaBN | Running statistics only | 0 | 0.8313 | 0.7113 |
| ConvLoRA + AdaBN | Running statistics adapt | 54,870 | 0.770088 | 0.626132 |
| ConvLoRA + AdaBN, detached pseudo-labels | Running statistics adapt | 54,870 | 0.768895 | 0.624557 |
| ConvLoRA + frozen BN | Running statistics frozen | 54,870 | 0.8355 | 0.7175 |
| ConvLoRA 3x3-only + frozen BN | Running statistics frozen | 52,182 | 0.8395 | 0.7233 |

The source-only result and the BN-only result use the 251-image target-evaluation set for reporting. The ConvLoRA experiments use the same separate target-evaluation set after adaptation. Target labels are not used for adaptation or checkpoint selection.

### BN-Only Diagnostic

The BN-only ablation froze all model parameters, disabled ConvLoRA and ESH, used no optimizer or pseudo-label loss, and fed only target adaptation images through the model in train mode to update BN running statistics. Five passes processed all 803 images. The target-evaluation Dice trajectory was 0.8258 at pass 0, then 0.8308, 0.8312, 0.8313, 0.8313, and 0.8313.

This modest improvement shows that BN-statistic adaptation alone does not explain the ConvLoRA failure. It does not establish that AdaBN is universally beneficial.

### ConvLoRA + AdaBN

The historical ConvLoRA + AdaBN run used an unlabeled adaptation subset, a 200-image consistency subset for checkpoint selection, and 5 adaptation epochs. Adaptation loss and consistency Dice were:

| Epoch | Loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5354 | 0.8117 |
| 2 | 0.5242 | 0.8102 |
| 3 | 0.5088 | 0.8107 |
| 4 | 0.4934 | 0.8202 |
| 5 | 0.4793 | 0.8214 |

The best consistency Dice was 0.8213968026 at epoch 5. Independent target evaluation yielded loss 0.297705, Dice 0.770088, IoU 0.626132, precision 0.651980, and recall 0.940451.

### Detached Pseudo-Labels

Experiment 2A changed only pseudo-label detachment. Its adaptation loss and consistency Dice were:

| Epoch | Loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5358 | 0.8114 |
| 2 | 0.5256 | 0.8079 |
| 3 | 0.5108 | 0.8064 |
| 4 | 0.4958 | 0.8163 |
| 5 | 0.4821 | 0.8168 |

The best consistency Dice was 0.8168 at epoch 5. Target evaluation yielded loss 0.297551, Dice 0.768895, IoU 0.624557, precision 0.654935, and recall 0.930868. Dice changed by approximately -0.001193 relative to the non-detached experiment. Pseudo-label detachment is therefore unlikely to be the dominant explanation for the severe negative transfer under this configuration, without ruling out all pseudo-label-related effects.

### ConvLoRA + Frozen BN

Freezing BN running statistics while retaining ConvLoRA produced the following adaptation trajectory:

| Epoch | Loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5565 | 0.8160 |
| 2 | 0.5271 | 0.8043 |
| 3 | 0.4899 | 0.8151 |
| 4 | 0.4481 | 0.8388 |
| 5 | 0.4118 | 0.8572 |

The best consistency Dice was 0.8572 at epoch 5. Independent target evaluation yielded loss 0.1975, Dice 0.8355, IoU 0.7175, precision 0.9068, and recall 0.7747.

Under the present experimental protocol, allowing BN running statistics to adapt jointly with ConvLoRA substantially degraded target-domain performance, whereas freezing BN running statistics removed this degradation. This pattern strongly implicates the interaction between ConvLoRA updates and BN running-statistic adaptation under this protocol. This is a protocol-specific conclusion, not a universal claim about AdaBN.

### ConvLoRA 3x3-Only + Frozen BN

The 3x3-only ablation excluded the 2x2 stride-2 downsampling convolutions while retaining 3x3 adapters in `[init_path, down1, down2, down3]`. Decoder and output convolutions remained unchanged. Rank 2, alpha 2, frozen BN running statistics, ESH/self-training, the 803/200 split, learning rate, optimizer, and 5-epoch protocol were preserved.

The adaptation trajectory was:

| Epoch | Loss | Consistency Dice |
| ---: | ---: | ---: |
| 1 | 0.5566 | 0.8146 |
| 2 | 0.5279 | 0.8021 |
| 3 | 0.4914 | 0.8192 |
| 4 | 0.4498 | 0.8350 |
| 5 | 0.4138 | 0.8552 |

The best consistency Dice was 0.8552 at epoch 5. Independent target evaluation yielded loss 0.1963, Dice 0.8395, IoU 0.7233, precision 0.9053, and recall 0.7825. Relative to full-encoder frozen-BN ConvLoRA, Dice improved by 0.0040 and IoU improved by 0.0058, while consistency Dice decreased from 0.8572 to 0.8552.

The result provides evidence that 2x2 downsampling adapters were not necessary to obtain the observed target-domain improvement in this protocol. It does not establish universal superiority of the 3x3-only placement.

## Metric Rationale And Evaluation Semantics

The original ConvLoRA paper used Surface Dice Score for CC359 evaluation. CC359 is a 3D brain MRI skull-stripping dataset in which accurate anatomical boundary delineation is important, making a surface-based metric appropriate for that evaluation setting. NIfTI and nibabel are data representation and loading choices; they are not the reason Surface Dice was selected.

The present project uses 2D dermoscopic ISIC skin-lesion segmentation. Binary Dice is the primary metric, with IoU, precision, and recall reported as complementary metrics. Metric choice is task-dependent; Surface Dice is not characterized as unsuitable for 2D segmentation.

Final target-evaluation metrics use labeled ISIC evaluation data. Adaptation checkpoint selection uses aggregated binary Dice on the unlabeled 200-image consistency subset and does not use target ground-truth labels. The project's binary Dice values are therefore not numerically equivalent to the paper's Surface Dice scores. This metric difference does not invalidate the internal ConvLoRA-versus-ConvMoRA comparison because both methods use the same evaluation protocol.

## Audit Findings And Remaining Concerns

The implementation audit found source checkpoint loading/freezing, ConvLoRA insertion scope, the `W + BA(alpha/r)` formulation, ESH freezing, target-label isolation, subset disjointness, and target-evaluation isolation to be correct.

The following concerns remain protocol or design observations:

- Consistency-based checkpoint selection is indirect because it does not use target-label Dice.
- Pseudo-labels were initially non-detached; the detached-pseudo-label experiment was run as a separate ablation.
- The `adabn_train_affine` field is not wired to runtime trainability. BN affine parameters remained frozen in the reported protocols.

## Preferred ConvLoRA Reference For The MoRA Comparison

The current preferred ConvLoRA reference is the 3x3-only frozen-BN variant:

- 3x3-only ConvLoRA in `[init_path, down1, down2, down3]`
- Rank 2
- Alpha 2
- Frozen BN running statistics
- ESH/self-training unchanged
- Independent target-evaluation Dice: 0.8395
- Independent target-evaluation IoU: 0.7233
- Independent target-evaluation precision: 0.9053
- Independent target-evaluation recall: 0.7825
- Trainable adapter parameters: 52,182

The preceding full-encoder frozen-BN ConvLoRA result remains a separate historical control with 54,870 trainable parameters and target Dice 0.8355. It is not replaced by the 3x3-only result.

## Planned ConvMoRA Constraint

ConvMoRA implementation has not begun. The planned MoRA update must use grouped row/column sharing with a square matrix under parameter matching. The comparison should change only the adapter parameterization while keeping the encoder insertion scope, 3x3-only adapter restriction, BN treatment, ESH, pseudo-label objective, target adaptation data, adaptation/consistency split, optimizer, training schedule, and evaluation protocol fixed wherever applicable.

Parameter matching must be calculated from flattened convolution dimensions. The reference first RGB-to-64 convolution requires special treatment because its flattened input dimension is only 9. If an equal-budget square MoRA dimension selected from parameter matching exceeds the smaller flattened dimension, a genuine high-rank/compressive MoRA mapping cannot be claimed. Exact parameter calculations and layer eligibility must be established before implementation.
