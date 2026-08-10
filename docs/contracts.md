# Data And Model Contracts

This document records the stable contracts established before implementing the baseline U-Net.

## Dataset Sample Contract

Every dataset adapter must expose a sample with these required keys:

- `image`
- `mask`
- `sample_id`
- `metadata`

### Image Contract

- Layout: `CHW`
- Shape: `(C, H, W)`
- Initial project setting: `C = 3` for RGB experiments
- Dtype at the common contract boundary: `float32`
- Value convention before optional normalization: `[0.0, 1.0]`
- Channel convention: RGB
- Spatial size: determined by `preprocessing.size`, not hard-coded in the model

### Mask Contract

- Layout: `1HW`
- Shape: `(1, H, W)` for current binary segmentation experiments
- Dtype at the common contract boundary: `float32`
- Value convention: binary values `{0.0, 1.0}`
- Resize interpolation: nearest-neighbor only
- Spatial dimensions must match the image after preprocessing

### Future Multi-Class Policy

- The common contract remains channel-first.
- Future semantic class-index masks should use a single semantic channel with class IDs encoded as integer labels before any loss-specific conversion.
- Any future one-hot expansion should happen in dedicated preprocessing or loss logic, not inside dataset-specific parsing.

### Sample ID And Metadata

- `sample_id` must be a stable string identifier suitable for logging, manifests, and prediction filenames.
- `metadata` may contain serializable dataset-specific information such as dataset name, version, split, original spatial shape, source paths, and acquisition metadata.
- `metadata` must not contain training state, optimizer state, or other experiment-global mutable objects.

## Model Input Contract

- Batch layout: `NCHW`
- Per-sample layout: `CHW`
- Dtype: `float32`
- Channel order: RGB for current experiments
- Spatial dimensions: `(H, W)` defined by `preprocessing.size`
- Value convention:
  - If normalization is disabled: values remain in `[0.0, 1.0]`
  - If normalization is enabled: values are normalized according to `preprocessing.normalization`
- The model must not perform dataset-specific preprocessing internally.

## Model Output Contract

- Required output key: `logits`
- Output shape for binary segmentation: `(N, 1, H, W)`
- Output dtype: floating-point tensor
- The model must return logits, not probabilities
- Sigmoid must not be applied inside the model
- Thresholding must occur outside the model
- Optional auxiliary outputs may be returned under `aux`

## Split Contract

- Train, validation, and test splits are explicit.
- Manifest-based splits are preferred.
- Split definitions belong to the dataset adapter/configuration layer, not to the trainer.
- ISIC 2016 is configured to preserve its explicit split manifests.

## Checkpoint Contract

- Checkpoint loading is always explicit.
- No checkpoint is selected automatically from prior runs.
- Dataset mismatch is disallowed by default.
- Model mismatch is disallowed by default.

## Future Conv-MoRA And Fisher Compatibility

- Models should expose stable module names for encoder, bottleneck, decoder, and head components.
- The trainer should interact with model/output contracts, not U-Net-specific internals.
- Future parameter-efficient modules must be attachable without changing the trainer API.
