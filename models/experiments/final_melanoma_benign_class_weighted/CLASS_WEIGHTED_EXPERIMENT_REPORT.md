# Final melanoma-vs-benign class-weighted experiment

This is a research/decision-support experiment. It is **not clinically validated**,
not a clinically calibrated probability, not diagnostic, and not deployment-ready.
Reported scores are AI prediction scores / probability estimates.

## 1. Objective

Test whether moderate class weighting improves the frozen 83.4% EfficientNetB0 +
MobileNetV2 baseline on `final_melanoma_benign`, keeping every other setting identical.

## 2. Dataset description

Frozen dataset at `dataset/processed/final_melanoma_benign/`.
Melanoma = clean DICM melanoma. Benign = clean DICM benign + ISIC 2020 named benign.
No images were added, removed, or reshuffled.

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

BGR→RGB; downscale max side 1024 if larger; DullRazor hair removal; light denoise;
resize 224; /255; ImageNet preprocess inside the model. No CLAHE. No lesion crop.
Read-only reuse of the frozen baseline JPEG cache (identical files; no writes).

## 8. Augmentation

Train only via `get_tf_data_augmentation_fn()`. Validation and test were not augmented.
Augmentation is identical to the frozen 83.4% baseline.

## 9. Architecture

EfficientNetB0 and MobileNetV2, ImageNet init, shared head:
GAP → Dropout → Dense(256) → BN → Dropout → Dense(1, sigmoid) named `malignant_prob`.

## 10. Training strategy

Three-stage fine-tune: frozen head 1e-3 / partial unfreeze 1e-4 / final 1e-5.
Early stopping and checkpoints on validation PR-AUC only. No oversampling.
The only change vs baseline is class weighting.

## 11. Loss function and class weights

BinaryCrossentropy(label_smoothing=0.05).
Formula: `weight(class) = N / (number_of_classes * class_count)`. Source: training set only.
N=11379, benign=6676, melanoma=4703.
benign_weight = 11379 / (2 × 6676) = 0.8522318754
melanoma_weight = 11379 / (2 × 4703) = 1.2097597278

## 12. Random seeds

Python/NumPy/TensorFlow/dataset seed = 42.

## 13. EfficientNet validation results

{
  "threshold": 0.45,
  "accuracy": 0.7940935192780968,
  "precision": 0.6930763740185582,
  "recall": 0.9309683604985618,
  "sensitivity": 0.9309683604985618,
  "specificity": 0.6917562724014337,
  "f1": 0.7945990180032734,
  "auc": 0.9050605332701024,
  "pr_auc": 0.8803188698023192,
  "tp": 971,
  "fp": 430,
  "tn": 965,
  "fn": 72,
  "n_positive": 1043,
  "n_negative": 1395
}

## 14. MobileNet validation results

{
  "threshold": 0.45,
  "accuracy": 0.8199343724364233,
  "precision": 0.8296943231441049,
  "recall": 0.7286673058485139,
  "sensitivity": 0.7286673058485139,
  "specificity": 0.8881720430107527,
  "f1": 0.7759060745278203,
  "auc": 0.9003120307082204,
  "pr_auc": 0.877376833265685,
  "tp": 760,
  "fp": 156,
  "tn": 1239,
  "fn": 283,
  "n_positive": 1043,
  "n_negative": 1395
}

## 15. Ensemble validation results

{
  "threshold": 0.45,
  "accuracy": 0.8215750615258408,
  "precision": 0.761168384879725,
  "recall": 0.8494726749760306,
  "sensitivity": 0.8494726749760306,
  "specificity": 0.8007168458781362,
  "f1": 0.8028998640688718,
  "auc": 0.9091770705539919,
  "pr_auc": 0.886431162979093,
  "tp": 886,
  "fp": 278,
  "tn": 1117,
  "fn": 157,
  "n_positive": 1043,
  "n_negative": 1395
}

## 16. Selected ensemble weights

EfficientNet 0.50 / MobileNet 0.50
Rule: max validation ROC-AUC, then validation F1 at 0.50

## 17. Selected threshold

0.45 (max validation F1 on the predefined 0.20-0.70 grid; ties by ROC-AUC then closeness to 0.50)

## 18. Final frozen test results

Evaluated once after weights, ensemble, and threshold were frozen.

### A. EfficientNetB0
{
  "threshold": 0.45,
  "accuracy": 0.8005744768157571,
  "precision": 0.7038941954445261,
  "recall": 0.920268972142171,
  "sensitivity": 0.920268972142171,
  "specificity": 0.7113180515759312,
  "f1": 0.79766860949209,
  "auc": 0.9083583120704415,
  "pr_auc": 0.8765437615563043,
  "tp": 958,
  "fp": 403,
  "tn": 993,
  "fn": 83,
  "n_positive": 1041,
  "n_negative": 1396
}

### B. MobileNetV2
{
  "threshold": 0.45,
  "accuracy": 0.8178087812884695,
  "precision": 0.8442906574394463,
  "recall": 0.7031700288184438,
  "sensitivity": 0.7031700288184438,
  "specificity": 0.9032951289398281,
  "f1": 0.7672955974842768,
  "auc": 0.9035025969629159,
  "pr_auc": 0.8769546194280387,
  "tp": 732,
  "fp": 135,
  "tn": 1261,
  "fn": 309,
  "n_positive": 1041,
  "n_negative": 1396
}

### C. Final ensemble
{
  "threshold": 0.45,
  "accuracy": 0.8272466146901929,
  "precision": 0.7690972222222222,
  "recall": 0.851104707012488,
  "sensitivity": 0.851104707012488,
  "specificity": 0.8094555873925502,
  "f1": 0.8080255357957137,
  "auc": 0.9121319592963565,
  "pr_auc": 0.8857626022922268,
  "tp": 886,
  "fp": 266,
  "tn": 1130,
  "fn": 155,
  "n_positive": 1041,
  "n_negative": 1396
}

## 19. Confusion matrix (ensemble test)

TP 886 / TN 1130 / FP 266 / FN 155

## 20. ROC-AUC

Frozen test ensemble ROC-AUC 0.9121

## 21. PR-AUC

Frozen test ensemble PR-AUC 0.8858

## 22. Training time

EfficientNet 17485.9s; MobileNet 11271.4s; total 28926.6s.
Best epoch (val PR-AUC): EfficientNet 12; MobileNet 36.

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

## 24. Comparison with frozen 83.4% baseline

Baseline numbers were read from the frozen experiment files and the quoted reference.
The baseline was not rerun.

Quoted frozen reference (as specified):
Accuracy 0.834, Precision 0.801, Sensitivity 0.815,
Specificity 0.849, F1 0.808, ROC-AUC 0.913, PR-AUC 0.888.
CM TP=848 FP=211 TN=1185 FN=193.
Ensemble 0.70/0.30, threshold 0.50.

Full-precision comparison against frozen `training_summary.json`:

Metric         Baseline       Class-weighted  Delta
Accuracy       0.8342         0.8272         -0.0070
Sensitivity    0.8146         0.8511         +0.0365
Specificity    0.8489         0.8095         -0.0394
Precision      0.8008         0.7691         -0.0317
F1             0.8076         0.8080         +0.0004
ROC-AUC        0.9127         0.9121         -0.0005
PR-AUC         0.8885         0.8858         -0.0027

Baseline CM: TP 848 FP 211 TN 1185 FN 193
New CM:      TP 886 FP 266 TN 1130 FN 155

## 25. Interpretation

**trade-off, not a clear improvement**

Some metrics improved and others worsened; class weighting shifted the operating point.

Internal frozen-test comparison only. Not clinically validated. Scores are AI probability estimates, not calibrated clinical probabilities. Derm7pt was not used.

## 26. Limitations

- Two-source dataset; source confounding is possible.
- Missing DICM patient IDs remain.
- No probability calibration.
- Not clinically validated.
- Internal test comparison only; Derm7pt was not used for tuning or evaluation.

## 27. Exact model file paths

C:\Projects\major-project\models\experiments\final_melanoma_benign_class_weighted\efficientnetb0_class_weighted.keras
C:\Projects\major-project\models\experiments\final_melanoma_benign_class_weighted\mobilenetv2_class_weighted.keras

## 28. Reproducibility information

Script: `scripts/train_final_melanoma_benign_class_weighted.py`.
Config: `training_config.json`. Class weights: `class_weights.json`.
Production models, Exp2-C, old 92.28% result, Derm7pt external, clean DICM,
the 83.4% baseline, and frozen CSVs were not modified.
Test set was not used for tuning. No post-test tuning was performed.

VALIDATION RESULTS are in sections 13-17. FINAL FROZEN TEST RESULTS are in sections 18-21.
