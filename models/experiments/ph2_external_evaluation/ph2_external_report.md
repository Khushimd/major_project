# PH2 external evaluation of two frozen models

TRAINING ON PH2: NO
THRESHOLD TUNING ON PH2: NO
ENSEMBLE-WEIGHT TUNING ON PH2: NO
CALIBRATION ON PH2: NO
MODELS MODIFIED: NO

Scores are **model-predicted melanoma scores / probability estimates**, not calibrated clinical probabilities.
This is **not** clinical validation.

PH2 was used previously in this project. This run is an external evaluation on the **fixed 200-case PH2 cohort**, not a claim that PH2 is a completely untouched dataset.

## Cohort

- Strict melanoma: 40
- Strict benign melanocytic nevi: 160 (80 common + 80 atypical)
- Total: 200
- Same cases for both models

## Old 92% production model (Exp2-C)

- EfficientNet weight: 0.75
- MobileNet weight: 0.25
- Threshold: 0.30 (frozen; not searched on PH2)
- Files: `C:\Projects\major-project\models\experiments\internal_optimization\exp2_loss_comparison\exp2c_bce_smooth\efficientnetb0_bce_smooth.keras` and `C:\Projects\major-project\models\experiments\internal_optimization\exp2_loss_comparison\exp2c_bce_smooth\mobilenetv2_bce_smooth.keras`

Historical internal result (~92.28% accuracy) used DICM class=1 and is **not** a pure melanoma-vs-benign number. Quoted only as historical internal reference.

{
  "threshold": 0.3,
  "accuracy": 0.205,
  "precision": 0.20100502512562815,
  "recall": 1.0,
  "sensitivity": 1.0,
  "specificity": 0.00625,
  "f1": 0.33472803347280333,
  "auc": 0.65734375,
  "pr_auc": 0.43084022554005486,
  "tp": 40,
  "fp": 159,
  "tn": 1,
  "fn": 0,
  "n_positive": 40,
  "n_negative": 160
}

CM TP=40 TN=1 FP=159 FN=0

## New class-weighted model

- EfficientNet weight: 0.50 (validation-selected)
- MobileNet weight: 0.50 (validation-selected)
- Threshold: 0.45 (validation-selected; not searched on PH2)
- Files: `C:\Projects\major-project\models\experiments\final_melanoma_benign_class_weighted\efficientnetb0_class_weighted.keras` and `C:\Projects\major-project\models\experiments\final_melanoma_benign_class_weighted\mobilenetv2_class_weighted.keras`

{
  "threshold": 0.45,
  "accuracy": 0.265,
  "precision": 0.21390374331550802,
  "recall": 1.0,
  "sensitivity": 1.0,
  "specificity": 0.08125,
  "f1": 0.3524229074889868,
  "auc": 0.834375,
  "pr_auc": 0.630438981058296,
  "tp": 40,
  "fp": 147,
  "tn": 13,
  "fn": 0,
  "n_positive": 40,
  "n_negative": 160
}

CM TP=40 TN=13 FP=147 FN=0

## Comparison (same 200 PH2 cases)

| Metric | Old 92% model | Class-weighted model |
|---|---:|---:|
| Accuracy | 0.2050 | 0.2650 |
| Sensitivity | 1.0000 | 1.0000 |
| Specificity | 0.0063 | 0.0813 |
| Precision | 0.2010 | 0.2139 |
| F1 | 0.3347 | 0.3524 |
| ROC-AUC | 0.6573 | 0.8344 |
| PR-AUC | 0.4308 | 0.6304 |
| TP | 40 | 40 |
| TN | 1 | 13 |
| FP | 159 | 147 |
| FN | 0 | 0 |
| Predicted melanoma | 199 | 187 |
| Predicted benign | 1 | 13 |

No model is ranked as best. Differences are observed values on this fixed PH2 cohort.

## Score distributions

See `score_distribution_summary.csv`. Values are model-predicted melanoma scores.

## Observations

Old model sensitivity 1.000 vs class-weighted 1.000; specificity 0.006 vs 0.081. False positives: old 159/160, class-weighted 147/160. Small PH2 melanoma n=40; treat metric differences as descriptive, not a ranking.

## Limitations

- PH2 was previously used in this project.
- 200 cases, 40 melanoma: metric variance is high.
- Not clinically validated. Scores are not calibrated clinical probabilities.
- Models were not changed based on these PH2 results.

## Reproducibility

{
  "started_utc": "2026-09-29T21:43:27.612975+00:00",
  "finished_utc": "2026-09-29T21:46:17.350917+00:00",
  "preprocessing": "process_for_model hair+denoise+224+/255; max-side 1024 if needed; no CLAHE; no crop",
  "environment": {
    "python": "3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]",
    "platform": "Windows-11-10.0.26200-SP0",
    "tensorflow": "2.21.0",
    "numpy": "2.5.3",
    "pandas": "3.0.5"
  }
}

