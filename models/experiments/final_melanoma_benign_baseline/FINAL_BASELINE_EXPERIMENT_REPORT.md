# Final melanoma-vs-benign baseline experiment

This is a research/decision-support experiment. It is **not clinically validated**,
not a clinically calibrated probability, not diagnostic, and not deployment-ready.
Reported scores are AI prediction scores / probability estimates.

## 1. Objective

Train one controlled EfficientNetB0 + MobileNetV2 baseline on the frozen
`final_melanoma_benign` dataset, select ensemble weights and threshold on
validation only, then evaluate the frozen test set once.

## 2. Dataset description

Frozen dataset at `dataset/processed/final_melanoma_benign/`.
Melanoma = clean DICM melanoma. Benign = clean DICM benign + ISIC 2020 named benign.
ISIC 2020 melanoma was not added. No images were added or removed for this run.

## 3. Dataset source breakdown

See the frozen `dataset_report.json` in the dataset directory.
DICM melanoma 6787, DICM benign 4586, ISIC 2020 benign 4881.

## 4. Class distribution

Total 16254; melanoma 6787 (41.8%); benign 9467 (58.2%); ratio 1:1.395.

## 5. Frozen train/validation/test split

Train 11379 (4703 / 6676). Validation 2438 (1043 / 1395). Test 2437 (1041 / 1396).
Grouping: patient → lesion → image. Seed 42. Splits were not recreated.

## 6. Leakage audit summary

Previous construction report: patient/lesion/image overlap across splits = 0.
This training run did not modify splits.

## 7. Preprocessing

BGR→RGB; downscale max side 1024 if larger (required for ~30MP ISIC 2020 images);
DullRazor hair removal; light denoise; resize 224; /255; ImageNet preprocess inside the model.
No CLAHE. No lesion crop. Same cache used for train, validation, and test.

## 8. Augmentation

Train only via `get_tf_data_augmentation_fn()`: flip, small rotation/zoom/translation,
brightness/contrast, mild saturation/hue, light noise, mild blur, occasional synthetic hair.
Validation and test were not augmented.

## 9. Architecture

EfficientNetB0 and MobileNetV2, ImageNet init, shared head:
GAP → Dropout → Dense(256) → BN → Dropout → Dense(1, sigmoid) named `malignant_prob`.

## 10. Training strategy

Three-stage fine-tune: frozen head 1e-3 / partial unfreeze 1e-4 / final 1e-5.
Early stopping and checkpoints on validation PR-AUC only. No class weights. No oversampling.

## 11. Loss function

BinaryCrossentropy(label_smoothing=0.05). Focal loss was not used.

## 12. Random seeds

Python/NumPy/TensorFlow/dataset seed = 42.

## 13. EfficientNet validation results

{
  "threshold": 0.5,
  "accuracy": 0.8096800656275636,
  "precision": 0.7367130008176614,
  "recall": 0.8638542665388304,
  "sensitivity": 0.8638542665388304,
  "specificity": 0.7691756272401433,
  "f1": 0.795233892321271,
  "auc": 0.9093671068773905,
  "pr_auc": 0.8863814155428249,
  "tp": 901,
  "fp": 322,
  "tn": 1073,
  "fn": 142,
  "n_positive": 1043,
  "n_negative": 1395
}

## 14. MobileNet validation results

{
  "threshold": 0.5,
  "accuracy": 0.8162428219852338,
  "precision": 0.8471411901983664,
  "recall": 0.6960690316395014,
  "sensitivity": 0.6960690316395014,
  "specificity": 0.9060931899641577,
  "f1": 0.7642105263157895,
  "auc": 0.8967563239483569,
  "pr_auc": 0.8741194473812097,
  "tp": 726,
  "fp": 131,
  "tn": 1264,
  "fn": 317,
  "n_positive": 1043,
  "n_negative": 1395
}

## 15. Ensemble validation results

{
  "threshold": 0.5,
  "accuracy": 0.8264971287940935,
  "precision": 0.7828467153284672,
  "recall": 0.822627037392138,
  "sensitivity": 0.822627037392138,
  "specificity": 0.8293906810035843,
  "f1": 0.8022440392706872,
  "auc": 0.911757853173744,
  "pr_auc": 0.8893139033924969,
  "tp": 858,
  "fp": 238,
  "tn": 1157,
  "fn": 185,
  "n_positive": 1043,
  "n_negative": 1395
}

## 16. Selected ensemble weights

EfficientNet 0.70 / MobileNet 0.30
Rule: max validation ROC-AUC, then validation F1 at 0.50

## 17. Selected threshold

0.50 (max validation F1 on the predefined 0.20-0.70 grid; ties by ROC-AUC then closeness to 0.50)

## 18. Final frozen test results

Evaluated once after weights, ensemble, and threshold were frozen.
{
  "threshold": 0.5,
  "accuracy": 0.8342224045958145,
  "precision": 0.8007554296506137,
  "recall": 0.8146013448607109,
  "sensitivity": 0.8146013448607109,
  "specificity": 0.8488538681948424,
  "f1": 0.8076190476190476,
  "auc": 0.9126790142826079,
  "pr_auc": 0.8884757664753524,
  "tp": 848,
  "fp": 211,
  "tn": 1185,
  "fn": 193,
  "n_positive": 1041,
  "n_negative": 1396
}

## 19. Confusion matrix

TP 848 / TN 1185 / FP 211 / FN 193

## 20. ROC-AUC

Frozen test ROC-AUC 0.9127

## 21. PR-AUC

Frozen test PR-AUC 0.8885

## 22. Training time

EfficientNet 15898.8s; MobileNet 11245.6s; total 27274.6s.

## 23. Hardware/environment

{
  "python": "3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.26200-SP0",
  "tensorflow": "2.21.0",
  "numpy": "2.5.3",
  "pandas": "3.0.5",
  "seed": 42,
  "gpu": []
}

## 24. Limitations

- Two-source dataset; source confounding is possible.
- Missing DICM patient IDs remain.
- No probability calibration.
- Not clinically validated.
- The previous ~92% experiment used a different (noisier) label definition and is not a like-for-like comparison.

## 25. Exact model file paths

C:\Projects\major-project\models\experiments\final_melanoma_benign_baseline\efficientnetb0_final_baseline.keras
C:\Projects\major-project\models\experiments\final_melanoma_benign_baseline\mobilenetv2_final_baseline.keras

## 26. Reproducibility information

Script: `scripts/train_final_melanoma_benign_baseline.py`. Config: `training_config.json`.
Production models, old 92% experiment, clean DICM experiment, and frozen CSVs were not modified.
Test set was not used for tuning. No post-test tuning was performed.

VALIDATION RESULTS are in sections 13-17. FINAL FROZEN TEST RESULTS are in sections 18-21.
