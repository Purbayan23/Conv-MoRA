# Expected Dataset Layout

This repository does not store raw medical data in Git.
The initial target dataset is ISIC 2016 binary skin lesion segmentation, but the dataset layer is designed so additional datasets can later be added through their own adapters and configuration without changing training code.

## Recommended Local Layout

```text
data/
└─ isic2016/
   ├─ raw/
   │  ├─ images/
   │  │  ├─ train/
   │  │  ├─ val/
   │  │  └─ test/
   │  └─ masks/
   │     ├─ train/
   │     ├─ val/
   │     └─ test/
   ├─ splits/
   │  ├─ train.csv
   │  ├─ val.csv
   │  └─ test.csv
   └─ metadata/
      └─ dataset_description.json
```

## Notes

- File names for each image and mask pair should align deterministically.
- Split manifests are the preferred source of truth even if directory names already indicate the split.
- For ISIC 2016, preserve the official split definition when applicable rather than creating a new random split implicitly.
- Masks may be stored with raw values such as `{0,255}`, but dataset adapters must convert them into the common binary contract `{0,1}`.
- Dataset-specific metadata may be stored here as needed, but the trainer should only consume the standardized sample contract.
- The actual parser and preprocessing pipeline are intentionally not implemented yet.

## Configurable Root

The dataset root can later be controlled with:

- `MEDSEG_DATA_ROOT`
- `configs/dataset/isic2016.yaml`

No dataset download, preprocessing, or manifest generation is performed by the current scaffold.
