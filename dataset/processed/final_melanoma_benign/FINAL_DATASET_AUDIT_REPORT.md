# Final melanoma-vs-benign research dataset audit

No model training was performed during this dataset-construction stage.

TRAINING PERFORMED: NO
PRODUCTION MODELS MODIFIED: NO
APPLICATION MODIFIED: NO
OLD DATASET MODIFIED: NO
CLEAN DICM DIRECTORY MODIFIED: NO

## 1. Dataset sources

- ISIC-DICM-17K clean melanoma from `dataset/processed/dicm_clean_melanoma_benign/`
- ISIC-DICM-17K clean benign from the same manifest
- ISIC 2020 strict named benign from `dataset/raw/train.csv` + JPEGs in `dataset/raw/train/`
- ISIC 2020 melanoma was not added

## 2. Exact inclusion criteria

- Melanoma: DICM `diagnosis==melanoma` AND `benign_malignant` in {malignant, indeterminate/malignant}
- DICM benign: diagnosis in the pre-specified named benign list AND `benign_malignant==benign`
- ISIC 2020 benign: named benign diagnosis AND `benign_malignant==benign`

## 3. Exact exclusion criteria

- Official DICM class was not used as a label
- ISIC 2020 `diagnosis=unknown` excluded
- ISIC 2020 atypical melanocytic proliferation excluded
- ISIC 2020 melanoma excluded (duplicate records vs DICM melanoma, ~581 IDs)
- ISIC 2020 strict-benign image IDs already present in DICM benign (532) kept as the DICM file only
- BCC/SCC/AK/AIMP/other ambiguous DICM diagnoses remain excluded by the clean DICM definition

## 4. Final counts by source and class

- DICM melanoma: 6787
- DICM benign: 4586
- ISIC 2020 benign added: 4881
- ISIC 2020 benign dropped as image-ID duplicates: 532
- ISIC 2020 extra copies dropped as byte-identical files: ['ISIC_4261232', 'ISIC_4364740', 'ISIC_7528783', 'ISIC_9645937']

## 5. Melanoma:benign ratio

6787:9467  (1:1.395)
Benign images were not downsampled to 1:1.

## 6. Image overlap results

- DICM melanoma vs ISIC benign image IDs: 0 (matches previous audit)
- DICM benign vs ISIC benign image IDs: 532 (matches previous audit; ISIC copies dropped)
- DICM melanoma vs DICM benign image IDs: 0
- Filename overlap DICM vs ISIC benign: 532
- Same-size same-hash extra copies dropped: ['ISIC_4261232', 'ISIC_4364740', 'ISIC_7528783', 'ISIC_9645937']
- Remaining same-hash duplicate files in the final pool: 0

## 7. Patient overlap results

{
  "dicm_melanoma_vs_dicm_benign": {
    "shared_patients": 107,
    "images_in_left": 212,
    "images_in_right": 154
  },
  "dicm_melanoma_vs_isic_benign": {
    "shared_patients": 179,
    "images_in_left": 255,
    "images_in_right": 671
  },
  "dicm_benign_vs_isic_benign": {
    "shared_patients": 363,
    "images_in_left": 545,
    "images_in_right": 3250
  }
}

Missing patient_id: {'dicm_melanoma': 4858, 'dicm_benign': 2200, 'isic_benign': 0}
Shared patients were not deleted. Their images are grouped into one split.

## 8. Lesion overlap results

ISIC 2020 has no lesion_id. DICM lesion IDs do not cross the new splits.
Unique non-empty lesions in the final pool: 4834

## 9. Source-related differences

See `source_bias_report.json`. These are potential source/domain confounding signals, not proof of model bias.
DICM rows have year/image_type/confirmation metadata; ISIC 2020 challenge CSV does not.
ISIC 2020 JPEGs are typically much larger native photographs than many DICM images.

## 10. Missing metadata limitations

- Many DICM rows lack patient_id and/or lesion_id; grouping falls back to lesion then image.
- ISIC 2020 has no lesion_id, confirmation type, year, or attribution in the local CSV.
- Patient overlap can exist even when image IDs do not.

## 11. Train/validation/test counts

- Train: 11379
- Validation: 2438
- Test: 2437
- Seed: 42

## 12. Class counts in each split

- Train melanoma/benign: 4703 / 6676
- Val melanoma/benign: 1043 / 1395
- Test melanoma/benign: 1041 / 1396

## 13. Leakage results

- patient overlap across splits: 0
- lesion overlap across splits: 0
- image overlap across splits: 0
- duplicate hashes across different paths: 0
- no image under both labels
- all referenced files exist
- no duplicate manifest rows

## 14. Exact random seed

42

## 15. Whether the dataset is safe to proceed to training

YES — internal leakage checks passed. Training was not started.
A later training run should treat this test split as frozen and should not use it for model selection.

## 16. Remaining limitations

- Two-source construction (DICM + ISIC 2020) remains a potential domain-confounding signal.
- Missing DICM patient IDs prevent complete patient grouping for those rows.
- ISIC lesion relationships cannot be audited.
- This dataset is not a clinical validation set.

No model training was performed during this dataset-construction stage.
