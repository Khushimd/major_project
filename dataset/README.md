# Dataset directory

## SIIM-ISIC 2020 (primary)

From the ISIC / Kaggle challenge page, download **only these two**:

| Download | Needed? |
|----------|---------|
| **Training Data → JPEG** (~23 GB) | Yes |
| **Training Ground Truth** (~2 MB) | Yes |
| Training metadata | No |
| DICOM / DICOM Corrected | No |
| Test Data (any) | No |
| Duplicate image list | No |

Expected layout:

```
dataset/raw/
├── train.csv   OR   ISIC_2020_Training_GroundTruth.csv
└── train/
    └── ISIC_*.jpg
```

The loader auto-detects `target`, `benign_malignant`, or `diagnosis` label columns.

Alternate image folders (auto-detected):

- `dataset/raw/jpeg/train/`
- `dataset/raw/images/`
- `dataset/raw/ISIC_2020_Training_JPEG/`

## HAM10000 (optional / future)

```
dataset/ham10000/
├── HAM10000_metadata.csv
└── images/            # or flat folder of *.jpg
```

Malignant mapping uses diagnosis code `mel` (see `config.DATASETS["ham10000"]`).

## Synthetic demo data

Created automatically under `dataset/processed/synthetic/` when you run:

```bash
python -m training.train --synthetic
# or
python scripts/initialize_models.py --quick-train
```
