# Stage 2 Reference Mapping

The paper-level target condition is **full encoder ConvLoRA plus AdaBN**. The executable reference is not fully consistent with that wording: its `constrained_lora_down3` branch enables BN-related parameters but does not call the LoRA replacement routine, while `test.py`'s `lora:down3` branch is the closest executable four-stage insertion (`init_path`, `down1`, `down2`, `down3`). The separate `full_lora` branch has inconsistent model-object flow and is not used as the mapping here.

This project therefore uses the most defensible executable mapping: ConvLoRA is inserted into all four encoder stages, base convolution and BN affine parameters are frozen, and the model remains in train mode so BN running statistics can adapt. The ESH is frozen and kept in evaluation mode. Target masks are loaded only by the target-evaluation command, never by adaptation.

The reference selects checkpoints with surface Dice, but this repository intentionally has no surface-distance dependency. Stage 2 uses aggregated binary consistency Dice for adaptation checkpoint selection and reports that substitution in checkpoint metadata. ISIC preprocessing uses the existing RGB 256x256 pipeline rather than the reference CC359 NIfTI-slice preprocessing.
