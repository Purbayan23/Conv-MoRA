"""Structured configuration schema for the research scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    """Repository-level metadata."""

    name: str = "conv-mora-seg"
    package: str = "medseg"
    task: str = "binary_medical_image_segmentation"


@dataclass
class SplitConfig:
    """Explicit definition for one dataset split."""

    name: str = "train"
    images_dir: str = "./data/isic2016/raw/images/train"
    masks_dir: str = "./data/isic2016/raw/masks/train"
    manifest_path: str | None = "./data/isic2016/splits/train.csv"
    definition_source: str = "official"


@dataclass
class DatasetSplitsConfig:
    """Split bundle for train, validation, and test."""

    train: SplitConfig = field(default_factory=SplitConfig)
    val: SplitConfig = field(
        default_factory=lambda: SplitConfig(
            name="val",
            images_dir="./data/isic2016/raw/images/val",
            masks_dir="./data/isic2016/raw/masks/val",
            manifest_path="./data/isic2016/splits/val.csv",
            definition_source="official",
        )
    )
    test: SplitConfig = field(
        default_factory=lambda: SplitConfig(
            name="test",
            images_dir="./data/isic2016/raw/images/test",
            masks_dir="./data/isic2016/raw/masks/test",
            manifest_path="./data/isic2016/splits/test.csv",
            definition_source="official",
        )
    )


@dataclass
class DataLoaderConfig:
    """Dataloader defaults owned by the dataset configuration."""

    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False


@dataclass
class DatasetConfig:
    """Dataset selection and dataset-specific behavior."""

    name: str = "isic2016"
    adapter: str = "isic2016"
    task: str = "binary_segmentation"
    version: str = "2016"
    split_definition_name: str = "official_isic2016_2016"
    image_color_space: str = "rgb"
    input_channels: int = 3
    num_classes: int = 1
    raw_mask_values: list[int] = field(default_factory=lambda: [0, 255])
    common_mask_values: list[int] = field(default_factory=lambda: [0, 1])
    sample_id_format: str = "filename_stem"
    root: str = "./data/isic2016"
    train_val_ratio: float = 0.8
    split_seed: int = 42
    metadata_keys: list[str] = field(
        default_factory=lambda: [
            "dataset_name",
            "dataset_version",
            "split",
            "source_image_path",
            "source_mask_path",
            "original_spatial_shape",
            "split_manifest",
        ]
    )
    splits: DatasetSplitsConfig = field(default_factory=DatasetSplitsConfig)
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


@dataclass
class ImageSizePolicyConfig:
    """Policy for resizing or otherwise standardizing spatial resolution."""

    name: str = "fixed_size"
    height: int = 256
    width: int = 256
    require_spatial_divisible_by: int = 16
    enforce_before_model: bool = True
    preserve_aspect_ratio: bool = False


@dataclass
class InterpolationConfig:
    """Interpolation policy for image and mask resizing."""

    image: str = "bilinear"
    mask: str = "nearest"


@dataclass
class NormalizationConfig:
    """Image normalization policy."""

    enabled: bool = False
    mode: str = "none"
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    input_range: list[float] = field(default_factory=lambda: [0.0, 1.0])


@dataclass
class PreprocessingConfig:
    """Dataset-agnostic preprocessing that standardizes model inputs."""

    name: str = "default_binary_preprocessing"
    image_dtype: str = "float32"
    mask_dtype: str = "float32"
    image_layout: str = "CHW"
    mask_layout: str = "1HW"
    channel_order: str = "rgb"
    scale_uint8_to_unit_interval: bool = True
    size: ImageSizePolicyConfig = field(default_factory=ImageSizePolicyConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class AugmentationPolicyConfig:
    """Rules that keep augmentation safe for segmentation masks."""

    shared_spatial_parameters: bool = True
    image_interpolation: str = "bilinear"
    mask_interpolation: str = "nearest"
    apply_image_only_transforms_to_mask: bool = False


@dataclass
class AugmentationStageConfig:
    """Train- or eval-stage augmentation plan."""

    enabled: bool = True
    allow_random: bool = False
    spatial: list[str] = field(default_factory=list)
    image_only: list[str] = field(default_factory=list)


@dataclass
class AugmentationConfig:
    """Augmentation settings separated from preprocessing."""

    name: str = "default_binary_augmentation"
    policy: AugmentationPolicyConfig = field(default_factory=AugmentationPolicyConfig)
    train: AugmentationStageConfig = field(
        default_factory=lambda: AugmentationStageConfig(
            enabled=True,
            allow_random=True,
            spatial=["random_horizontal_flip", "random_vertical_flip", "random_rotate_90"],
            image_only=[],
        )
    )
    eval: AugmentationStageConfig = field(
        default_factory=lambda: AugmentationStageConfig(
            enabled=True,
            allow_random=False,
            spatial=[],
            image_only=[],
        )
    )


@dataclass
class ConvBlockConfig:
    """Replaceable convolution block settings."""

    name: str = "double_conv"
    kernel_size: int = 3
    repeats: int = 2
    bias: bool = True
    dropout: float = 0.0


@dataclass
class EncoderBlockConfig:
    """Replaceable encoder stage settings."""

    name: str = "unet_encoder_stage"
    downsample: str = "max_pool"
    skip_connections: bool = True


@dataclass
class BottleneckBlockConfig:
    """Replaceable bottleneck stage settings."""

    name: str = "unet_bottleneck_stage"
    dropout: float = 0.0


@dataclass
class DecoderBlockConfig:
    """Replaceable decoder stage settings."""

    name: str = "unet_decoder_stage"
    upsample_mode: str = "transposed_conv"
    merge_mode: str = "concat"


@dataclass
class NormConfig:
    """Normalization layer settings."""

    name: str = "none"
    eps: float = 1.0e-05
    affine: bool = True


@dataclass
class ActivationConfig:
    """Activation layer settings."""

    name: str = "relu"
    inplace: bool = True


@dataclass
class BlocksConfig:
    """Collection of replaceable U-Net block settings."""

    conv: ConvBlockConfig = field(default_factory=ConvBlockConfig)
    encoder: EncoderBlockConfig = field(default_factory=EncoderBlockConfig)
    bottleneck: BottleneckBlockConfig = field(default_factory=BottleneckBlockConfig)
    decoder: DecoderBlockConfig = field(default_factory=DecoderBlockConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)


@dataclass
class ModelConfig:
    """Model family and architectural hyperparameters."""

    name: str = "unet"
    variant: str = "ronneberger2015"
    architecture: str = "unet_ronneberger2015"
    implemented: bool = False
    in_channels: int = 3
    out_channels: int = 1
    encoder_depth: int = 4
    feature_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    padding: str = "valid"
    convolution_block: str = "double_conv"
    encoder_block: str = "unet_encoder_stage"
    bottleneck_block: str = "unet_bottleneck_stage"
    decoder_block: str = "unet_decoder_stage"
    normalization: str = "none"
    activation: str = "relu"
    head: str = "binary_segmentation_head"
    stable_module_naming: bool = True


@dataclass
class HeadConfig:
    """Segmentation head settings."""

    name: str = "binary_segmentation_head"
    out_channels: int = 1
    activation: str = "none"
    threshold: float = 0.5


@dataclass
class LossConfig:
    """Loss selection and weighting."""

    name: str = "bce_dice"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    from_logits: bool = True
    smooth: float = 1.0


@dataclass
class MetricsConfig:
    """Metric selection for binary segmentation."""

    names: list[str] = field(default_factory=lambda: ["dice", "iou"])
    threshold: float = 0.5
    from_logits: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters."""

    name: str = "adam"
    lr: float = 1.0e-04
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    """Learning-rate scheduler settings."""

    name: str = "none"
    interval: str = "epoch"


@dataclass
class CheckpointConfig:
    """Explicit checkpoint loading policy."""

    load_enabled: bool = False
    load_path: str | None = None
    strict: bool = True
    allow_dataset_mismatch: bool = False
    allow_model_mismatch: bool = False


@dataclass
class RuntimeConfig:
    """Runtime controls for local and Colab execution."""

    profile_name: str = "local"
    deterministic: bool = True
    device: str = "auto"
    amp: bool = False
    compile_model: bool = False
    matmul_precision: str = "high"
    num_workers: int | None = None
    pin_memory: bool | None = None
    prefetch_factor: int | None = 2


@dataclass
class ExperimentConfig:
    """Experiment naming, tracking, and checkpoint saving policy."""

    name: str = "baseline_isic2016_unet"
    output_root: str = "./outputs"
    tags: list[str] = field(default_factory=lambda: ["baseline", "unet", "isic2016"])
    save_top_k: int = 1
    monitor: str = "val_dice"
    monitor_mode: str = "max"
    max_epochs: int = 50
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 1
    manifest_filename: str = "run_manifest.json"


@dataclass
class AppConfig:
    """Top-level configuration object for the entire project."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    blocks: BlocksConfig = field(default_factory=BlocksConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
