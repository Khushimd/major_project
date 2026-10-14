# Frozen old Exp2-C model — Derm7pt external validation

"The previously frozen production model was evaluated on the frozen Derm7pt external cohort without retraining, fine-tuning, threshold optimization, ensemble-weight optimization, or probability calibration."

Scores below are model-predicted melanoma probability/score estimates. They are not calibrated clinical probabilities.

## 1. Objective

Obtain one unbiased external-validation result for the already-frozen old Exp2-C ensemble on the already-frozen Derm7pt dermoscopic cohort.

## 2. Frozen model specification

- EfficientNetB0 + MobileNetV2 soft-voting ensemble
- EfficientNet weight: 0.75
- MobileNet weight: 0.25
- Operating threshold: 0.30 (frozen; not searched on Derm7pt)
- EfficientNet: `C:\Projects\major-project\models\experiments\internal_optimization\exp2_loss_comparison\exp2c_bce_smooth\efficientnetb0_bce_smooth.keras`
- EfficientNet SHA-256: `02755e9c69e9f885b71696a278f807c9a93c29b88d59f7d252002f8bbbb62f87`
- MobileNet: `C:\Projects\major-project\models\experiments\internal_optimization\exp2_loss_comparison\exp2c_bce_smooth\mobilenetv2_bce_smooth.keras`
- MobileNet SHA-256: `ad049149423ad1a5a73f9b1bd6dabd0765968ae532472b8afd5757eb7f33146f`
- Preprocessing: existing `ImagePreprocessor.process_for_model` (hair removal + light denoise + resize 224 + /255; no CLAHE; no lesion crop; ImageNet preprocess inside the saved models)

## 3. OLD INTERNAL TEST (frozen, not recomputed)

This is **not** the Derm7pt result.

- Test images: 2,473
- Accuracy: 92.28%
- Sensitivity: 95.24%
- Specificity: 88.86%
- Precision: 90.78%
- F1: 92.96%
- ROC-AUC: 97.99%
- PR-AUC: 98.36%
- TP 1,261 / TN 1,021 / FP 128 / FN 63

This historical internal result used DICM `class=1` as the positive class, which is NOT equivalent to a pure melanoma label. Therefore 92.28% is not a clean melanoma-vs-benign result.

## 4. DERM7PT EXTERNAL VALIDATION

- Evaluated cases: 987
- Melanoma: 252
- Non-melanoma: 735
- Excluded (not scored): 24
- Image used: dermoscopic only

- Accuracy: 0.2766 (27.66%)
- Sensitivity / Recall: 0.9960 (99.60%)
- Specificity: 0.0299 (2.99%)
- Precision: 0.2604 (26.04%)
- F1-score: 0.4128
- ROC-AUC: 0.7000 (continuous ensemble scores)
- PR-AUC: 0.4775 (continuous ensemble scores)

- TP: 251
- TN: 22
- FP: 713
- FN: 1

Do not combine this external result with the old internal 92.28% numbers.

## 5. Confusion matrix

```
                 Predicted Benign    Predicted Melanoma
Actual Benign    22                 713
Actual Melanoma  1                  251
```

Here “Benign” is the frozen Derm7pt non-melanoma class (nevi, BCC, SK, and other clearly non-melanoma diagnoses), not a claim that every negative case is clinically benign.

## 6. Diagnosis distribution (descriptive)

These rows are counts and mean model-predicted melanoma scores. Accuracy is not reported for small categories.

| Diagnosis | n | Predicted melanoma | Predicted non-melanoma | Mean ensemble score | Note |
|---|---:|---:|---:|---:|---|
| basal cell carcinoma | 42 | 42 | 0 | 0.7721 | descriptive_counts_and_mean_score_only |
| blue nevus | 28 | 28 | 0 | 0.6274 | descriptive_only_sample_too_small |
| clark nevus | 399 | 384 | 15 | 0.5424 | descriptive_counts_and_mean_score_only |
| combined nevus | 13 | 13 | 0 | 0.6491 | descriptive_only_sample_too_small |
| congenital nevus | 17 | 17 | 0 | 0.6355 | descriptive_only_sample_too_small |
| dermal nevus | 33 | 32 | 1 | 0.5942 | descriptive_counts_and_mean_score_only |
| dermatofibroma | 20 | 19 | 1 | 0.4804 | descriptive_only_sample_too_small |
| lentigo | 24 | 23 | 1 | 0.4925 | descriptive_only_sample_too_small |
| melanoma (in situ) | 64 | 63 | 1 | 0.6963 | descriptive_counts_and_mean_score_only |
| melanoma (less than 0.76 mm) | 102 | 102 | 0 | 0.7392 | descriptive_counts_and_mean_score_only |
| melanoma (0.76 to 1.5 mm) | 53 | 53 | 0 | 0.8099 | descriptive_counts_and_mean_score_only |
| melanoma (more than 1.5 mm) | 28 | 28 | 0 | 0.8063 | descriptive_only_sample_too_small |
| melanoma metastasis | 4 | 4 | 0 | 0.4502 | descriptive_only_sample_too_small |
| recurrent nevus | 6 | 6 | 0 | 0.5154 | descriptive_only_sample_too_small |
| reed or spitz nevus | 79 | 79 | 0 | 0.6332 | descriptive_counts_and_mean_score_only |
| seborrheic keratosis | 45 | 42 | 3 | 0.7076 | descriptive_counts_and_mean_score_only |
| vascular lesion | 29 | 28 | 1 | 0.6260 | descriptive_only_sample_too_small |
| melanoma | 1 | 1 | 0 | 0.5101 | descriptive_only_sample_too_small |

## 7. Limitations

- External dataset shift: Derm7pt cameras, population, and labels differ from the old development set.
- The old model was trained with DICM `class=1` as the positive class, not pure melanoma.
- Derm7pt negatives include BCC and other non-melanoma diagnoses.
- Patient-level independence could not be completely proven.
- Threshold 0.30 was frozen before this evaluation and was not optimized on Derm7pt.
- This is not a clinical validation study.

## 8. Reproducibility

- Evaluation UTC: 2026-09-29T11:11:44.737177+00:00
- Python: 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
- TensorFlow: 2.21.0
- pandas: 3.0.5
- Script: `scripts/eval_old_model_derm7pt_external.py`
- Frozen cohort: `models/experiments/old_model_derm7pt_external/external_manifest.csv`
- Training performed: no
- Threshold search: no
- Ensemble-weight search: no
- Calibration: no

## 9. STOP

External validation complete. No further experiment started.

