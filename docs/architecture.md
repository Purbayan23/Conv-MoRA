# Architecture Notes

This scaffold is organized around replaceable components rather than a single monolithic training script.

## Principles

- Keep the initial baseline faithful and simple.
- Keep training code model-agnostic.
- Make model blocks replaceable through configuration and builders.
- Isolate dataset-specific parsing, split definitions, mask conversion, and metadata inside dataset adapters.
- Separate preprocessing from augmentation so model inputs stay standardized.
- Separate architecture changes from training-time adaptation logic.
- Preserve reproducibility through explicit runtime and experiment settings.
- Treat each dataset/model combination as an independent experiment unless transfer learning is explicitly requested in a future study.

## Extension Strategy

Future research modules such as Conv-MoRA or Fisher-guided adaptation should be introduced through:

- model block replacement in `src/medseg/models/blocks/`
- optional model augmentation hooks in `src/medseg/models/extensions/`
- parameter grouping and selective fine-tuning logic in `src/medseg/optimization/`

This separation keeps the trainer stable while the research surface evolves.

## Explicit Non-Goals

- No continual-learning assumptions are built into the scaffold.
- No checkpoint is reused across datasets unless a future experiment explicitly requests it.
