# Medical Image Segmentation Research Scaffold

This repository is a research-oriented PyTorch scaffold for medical image segmentation.
Its first scientific objective is a faithful reproduction of the original U-Net from Ronneberger et al. (2015), initially evaluated on ISIC 2016 binary skin lesion segmentation.

The scaffold is intentionally being built in stages.
At the current stage, the emphasis is on explicit dataset contracts, configuration-driven experiments, and a model-agnostic training architecture that can support future Conv-MoRA and Fisher-guided experiments without redesigning the trainer.

## Research Scope

- Independent experiments across multiple medical segmentation datasets
- Reusable dataset, training, validation, testing, metric, and experiment infrastructure
- Dataset-specific parsing, preprocessing, split definitions, and metadata isolated in dataset adapters
- Centralized YAML/dataclass configuration for dataset, preprocessing, augmentation, model, optimization, runtime, and checkpoint policy
- Explicit experiment identity so results can always be traced back to dataset, model, split definition, preprocessing, augmentation, and seed

## Explicitly Out of Scope Right Now

- U-Net implementation
- Conv-MoRA implementation
- Fisher computation or selection
- LoRA or adapters
- Continual learning
- Implicit transfer learning or automatic checkpoint reuse across datasets
- Full training, validation, or inference execution logic

## Current Structure

- `configs/`: YAML configuration groups for dataset, preprocessing, augmentation, model, optimizer, runtime, checkpoint policy, and experiment tracking
- `src/medseg/`: research code organized around contracts and replaceable components
- `docs/`: architectural notes and explicit data/model contracts
- `tests/`: contract-focused unit tests for configuration, dataset adapters, transforms, model I/O contracts, and experiment identity
- `data/`: expected dataset layout documentation only
- `outputs/`: run artifacts such as manifests, logs, checkpoints, figures, and predictions

## Getting Started

1. Create an environment with Python 3.10 or newer.
2. Install the project in editable mode with `pip install -r requirements/base.txt`.
3. Optionally install development dependencies with `pip install -r requirements/dev.txt`.
4. Review [docs/contracts.md](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/docs/contracts.md), [configs/config.yaml](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/configs/config.yaml), and [data/README.md](C:/Users/user/OneDrive/Documents/ChatGPT/Conv-MoRA/data/README.md).

## Next Planned Step

The next implementation step is still the baseline U-Net and its building blocks.
The current scaffold stops before model implementation by design.
