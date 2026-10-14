# Exp2-C FROZEN

Status: **FROZEN**. Weights, threshold, ensemble, and test numbers below are locked.

Do **not** overwrite these `.keras` files. Do **not** replace production. Do **not** replace the 83.42% melanoma-vs-benign baseline. This is **not** a clean melanoma-vs-benign model and is **not** clinically validated.

## Frozen files

- EfficientNetB0: `efficientnetb0_bce_smooth.keras`
- SHA-256: `02755e9c69e9f885b71696a278f807c9a93c29b88d59f7d252002f8bbbb62f87`
- MobileNetV2: `mobilenetv2_bce_smooth.keras`
- SHA-256: `ad049149423ad1a5a73f9b1bd6dabd0765968ae532472b8afd5757eb7f33146f`
- Ensemble: 0.75 / 0.25
- Threshold: 0.30
- Loss: BCE + label smoothing 0.05

## Labels

- Class 0 = ISIC 2020 official benign (`benign_malignant=benign`). DICM official class=0 was not used.
- Class 1 = DICM official class=1. This is **not** `diagnosis == melanoma` only.

## Frozen internal test (as labeled)

2,473 images: 1,324 class 1, 1,149 class 0.

| Metric | Value |
|---|---|
| Accuracy | 92.28% |
| Sensitivity / recall | 95.24% |
| Specificity | 88.86% |
| Precision | 90.78% |
| F1 | 0.9296 |
| ROC-AUC | 0.9799 |
| PR-AUC | 0.9836 |
| TP / TN / FP / FN | 1261 / 1021 / 128 / 63 |

## Class 1 on that same frozen test, by diagnosis

These images were all scored as the **positive** class. Calling them melanoma counts as TP under 92.28%.

| Diagnosis | n | Called positive | Called negative | Mean score |
|---|---:|---:|---:|---:|
| melanoma | 1092 | 1035 | 57 | 0.773 |
| basal cell carcinoma | 132 | 127 | 5 | 0.654 |
| squamous cell carcinoma | 75 | 75 | 0 | 0.642 |
| melanoma metastasis | 10 | 10 | 0 | 0.788 |
| missing | 7 | 6 | 1 | 0.745 |
| AIMP | 4 | 4 | 0 | 0.445 |
| seborrheic keratosis | 3 | 3 | 0 | 0.441 |
| atypical melanocytic proliferation | 1 | 1 | 0 | 0.485 |
| **not diagnosis=melanoma** | **232** | **226** | **6** | **0.652** |

232 / 1324 class-1 test images are not `diagnosis=melanoma`. 226 of those 232 were counted as true positives in 92.28%.
