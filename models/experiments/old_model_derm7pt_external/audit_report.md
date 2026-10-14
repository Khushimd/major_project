# Derm7pt audit for the frozen old Exp2-C model

The old production model was **not** evaluated on Derm7pt in this step. This document freezes the external cohort only.

Statement for the later evaluation (not yet executed):

> The old production model was evaluated on Derm7pt without retraining, fine-tuning, threshold optimization, ensemble-weight optimization, or calibration.

That statement will apply only after Phase C inference is run against this frozen manifest.

## 1. Objective

Prepare an independent Derm7pt dermoscopic cohort for external validation of the already-frozen old Exp2-C ensemble (EfficientNetB0 0.75 + MobileNetV2 0.25, threshold 0.30).

## 2. Frozen model specification

- Model: EfficientNetB0 + MobileNetV2, soft-voting ensemble
- EfficientNet weight: 0.75
- MobileNet weight: 0.25
- Operating threshold: 0.30 (frozen; not to be retuned on Derm7pt)
- EfficientNet file: `C:\Projects\major-project\models\experiments\internal_optimization\exp2_loss_comparison\exp2c_bce_smooth\efficientnetb0_bce_smooth.keras`
- EfficientNet SHA-256: `02755e9c69e9f885b71696a278f807c9a93c29b88d59f7d252002f8bbbb62f87`
- MobileNet file: `C:\Projects\major-project\models\experiments\internal_optimization\exp2_loss_comparison\exp2c_bce_smooth\mobilenetv2_bce_smooth.keras`
- MobileNet SHA-256: `ad049149423ad1a5a73f9b1bd6dabd0765968ae532472b8afd5757eb7f33146f`
- Preprocessing to use later: existing `ImagePreprocessor.process_for_model`
- Scores are model-predicted melanoma probability/score estimates, not calibrated clinical probabilities

## 3. OLD INTERNAL TEST RESULT (frozen, not recomputed)

This is **not** the Derm7pt result. It is quoted from the existing frozen evaluation.

- Test images: 2473
- Accuracy: 92.28%
- Sensitivity: 95.24%
- Specificity: 88.86%
- Precision: 90.78%
- F1: 92.96%
- ROC-AUC: 97.99%
- PR-AUC: 98.36%
- TP 1261 / TN 1021 / FP 128 / FN 63

This internal result used DICM class=1 as the positive class and is not a clean melanoma-vs-benign result. It was not recomputed in this audit.

## 4. Derm7pt dataset

- Path: `C:\Projects\major-project\dataset\external\derm7pt`
- Rows in meta.csv: 1011
- Columns: case_num, diagnosis, seven_point_score, pigment_network, streaks, pigmentation, regression_structures, dots_and_globules, blue_whitish_veil, vascular_structures, level_of_diagnostic_difficulty, elevation, location, sex, management, clinic, derm, case_id, notes
- Diagnosis column: `diagnosis`
- Dermoscopic image column: `derm`
- Clinical image column: `clinic` (not used)
- Dermoscopic images found: 1011
- Missing dermoscopic images: 0
- Duplicate derm filenames: 0
- Duplicate derm SHA-256 values: 0

Official original index files were used. No new random split was created.

- Official train indexes: 413
- Official valid indexes: 203
- Official test indexes: 395

The later external evaluation cohort is **all eligible independent cases**, not only the official test split, because this model was not trained on Derm7pt. Official split membership is stored on every row.

### Diagnosis distribution (complete, inspected from meta.csv)

| Diagnosis | Count | Mapping |
|---|---:|---|
| clark nevus | 399 | negative_clearly_non_melanoma |
| melanoma (less than 0.76 mm) | 102 | positive_melanoma |
| reed or spitz nevus | 79 | negative_clearly_non_melanoma |
| melanoma (in situ) | 64 | positive_melanoma |
| melanoma (0.76 to 1.5 mm) | 53 | positive_melanoma |
| seborrheic keratosis | 45 | negative_clearly_non_melanoma |
| basal cell carcinoma | 42 | negative_clearly_non_melanoma |
| dermal nevus | 33 | negative_clearly_non_melanoma |
| vascular lesion | 29 | negative_clearly_non_melanoma |
| blue nevus | 28 | negative_clearly_non_melanoma |
| melanoma (more than 1.5 mm) | 28 | positive_melanoma |
| lentigo | 24 | negative_clearly_non_melanoma |
| dermatofibroma | 20 | negative_clearly_non_melanoma |
| congenital nevus | 17 | negative_clearly_non_melanoma |
| melanosis | 16 | excluded_ambiguous |
| combined nevus | 13 | negative_clearly_non_melanoma |
| miscellaneous | 8 | excluded_ambiguous |
| recurrent nevus | 6 | negative_clearly_non_melanoma |
| melanoma metastasis | 4 | positive_melanoma |
| melanoma | 1 | positive_melanoma |

### Missing values

| Column | Missing |
|---|---:|
| case_num | 0 |
| diagnosis | 0 |
| seven_point_score | 0 |
| pigment_network | 0 |
| streaks | 0 |
| pigmentation | 0 |
| regression_structures | 0 |
| dots_and_globules | 0 |
| blue_whitish_veil | 0 |
| vascular_structures | 0 |
| level_of_diagnostic_difficulty | 0 |
| elevation | 0 |
| location | 0 |
| sex | 0 |
| management | 0 |
| clinic | 0 |
| derm | 0 |
| case_id | 984 |
| notes | 1002 |

## 5. Label definition

Binary mapping for this external validation only:

Positive (1) = melanoma diagnoses:

- `melanoma`
- `melanoma (0.76 to 1.5 mm)`
- `melanoma (in situ)`
- `melanoma (less than 0.76 mm)`
- `melanoma (more than 1.5 mm)`
- `melanoma metastasis`

Negative (0) = clearly non-melanoma diagnoses:

- `basal cell carcinoma`
- `blue nevus`
- `clark nevus`
- `combined nevus`
- `congenital nevus`
- `dermal nevus`
- `dermatofibroma`
- `lentigo`
- `recurrent nevus`
- `reed or spitz nevus`
- `seborrheic keratosis`
- `vascular lesion`

Notes:

- Melanoma in situ is included as melanoma.
- All listed melanoma thickness categories are included as melanoma.
- Melanoma metastasis (n=4) is included as melanoma because it is a melanoma diagnosis; it is reported separately because it is not a primary cutaneous lesion.
- Melanosis is excluded: not melanoma and not automatically treated as non-melanoma/benign.
- Miscellaneous is excluded: unspecified mixed category.
- Basal cell carcinoma is non-melanoma (negative), not benign in the clinical sense.
- The old development labels used DICM official class=1 as the positive class, which is not a pure melanoma label.

## 6. Exclusion criteria

A case is excluded from the frozen evaluation cohort if any of the following apply:

- the dermoscopic image is missing
- the diagnosis is ambiguous (`melanosis`, `miscellaneous`, or any later unmapped label)
- a high-confidence overlap with old development data is found (exact filename/ID, exact SHA-256, or identical average-hash **and** identical file size)

Excluded / ambiguous diagnosis reasons:

- `melanosis` (n=16): ambiguous_not_melanoma_not_clearly_benign: melanosis is not a melanoma diagnosis and is not automatically assigned to the non-melanoma class
- `miscellaneous` (n=8): ambiguous_unspecified_mixed_category: miscellaneous is not a named diagnosis and cannot be defensibly assigned

No case was silently discarded. Every row is in `external_manifest.csv` with an inclusion flag and reason.

## 7. Overlap / leakage audit

Old development dataset: `dataset/processed/melanoma_benign` (n=16529; train 11611, val 2445, test 2473).

- Exact filename/ID match rows: 0
- Exact SHA-256 match rows: 0
- Same-size image matches: 0
- High-confidence same-size perceptual exact matches: 0
- Perceptual candidate rows (8x8 average hash, Hamming 0-6): 26099 across 494 Derm7pt cases
- Hamming 0 / 1 / 2 unique cases: 10 / included in 102 with Hamming <= 2
- Those Hamming-0 rows all had different file sizes, so they are hash collisions, not byte duplicates
- High-confidence duplicate Derm7pt cases excluded: 0
- Independent cases remaining: 987

8x8 average-hash Hamming <= 6 is a weak screen on this corpus. Those rows are reported only and were not used to drop cases.

Derm7pt is treated as an external cohort after filename, SHA-256, and high-confidence perceptual-hash checks against the frozen old development images. Patient-level independence is not fully proven.

### Patient / lesion identifiers

Patient-level overlap could not be completely assessed. Derm7pt has no ISIC-style patient_id. The sparse case_id field does not share a namespace with old patient/lesion/image identifiers. No identifier overlap was found among the nonempty Derm7pt case_id values. Identity was not inferred from sex, location, or other unrelated fields.

- Derm7pt case_id ∩ old patient_id: 0
- Derm7pt case_id ∩ old lesion_id: 0
- Derm7pt case_id ∩ old group_id: 0
- Derm7pt case_id ∩ old image_id: 0

## 8. External evaluation cohort (frozen)

- Total Derm7pt cases: 1011
- Eligible melanoma (label-mapped, image present): 252
- Eligible non-melanoma (label-mapped, image present): 735
- Excluded ambiguous diagnoses: 24
- Excluded for overlap: 0
- Excluded total: 24
- Frozen evaluation n: 987 (252 melanoma / 735 non-melanoma)

This manifest is now frozen. Do not tune weights or thresholds on it.

## 9. External performance

Not computed. Inference was not run.

## 10. Confusion matrix

Not computed. Inference was not run.

## 11. Diagnosis distribution

| Diagnosis | Total | Included | Excluded |
|---|---:|---:|---:|
| clark nevus | 399 | 399 | 0 |
| melanoma (less than 0.76 mm) | 102 | 102 | 0 |
| reed or spitz nevus | 79 | 79 | 0 |
| melanoma (in situ) | 64 | 64 | 0 |
| melanoma (0.76 to 1.5 mm) | 53 | 53 | 0 |
| seborrheic keratosis | 45 | 45 | 0 |
| basal cell carcinoma | 42 | 42 | 0 |
| dermal nevus | 33 | 33 | 0 |
| vascular lesion | 29 | 29 | 0 |
| blue nevus | 28 | 28 | 0 |
| melanoma (more than 1.5 mm) | 28 | 28 | 0 |
| lentigo | 24 | 24 | 0 |
| dermatofibroma | 20 | 20 | 0 |
| congenital nevus | 17 | 17 | 0 |
| melanosis | 16 | 0 | 16 |
| combined nevus | 13 | 13 | 0 |
| miscellaneous | 8 | 0 | 8 |
| recurrent nevus | 6 | 6 | 0 |
| melanoma metastasis | 4 | 4 | 0 |
| melanoma | 1 | 1 | 0 |

## 12. Limitations

- Derm7pt and the old development set come from different sources, cameras, and labeling protocols.
- The old model’s positive class is not pure melanoma (DICM `class=1`). Derm7pt negatives include BCC and other non-melanoma diagnoses.
- Patient-level overlap cannot be completely ruled out.
- 8×8 average-hash candidates with Hamming distance 1–6 were reported and not auto-excluded unless they also had exact SHA-256 or same-size exact hash.
- Official Derm7pt train/valid/test indexes were recorded but were not used to drop independent cases.
- No claim of clinical validation is made.

## 13. Reproducibility information

- Audit started (UTC): 2026-09-29T02:21:59.240842+00:00
- Audit finished (UTC): 2026-09-29T02:30:45.607179+00:00
- Python: 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
- Platform: Windows-11-10.0.26200-SP0
- pandas: 3.0.5
- Script: `scripts/audit_old_model_derm7pt_external.py`
- Outputs: `models/experiments/old_model_derm7pt_external/`

Reproduce the cohort by re-running the audit script. It only reads frozen files and writes this directory.

## 14. STOP

STOP: audit and cohort freeze completed. Model inference was not run.
