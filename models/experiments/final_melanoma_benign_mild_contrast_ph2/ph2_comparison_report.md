# PH2: frozen 83.42% baseline vs mild-contrast augmentation

TRAINING ON PH2: NO
THRESHOLD TUNING ON PH2: NO
ENSEMBLE-WEIGHT TUNING ON PH2: NO
CALIBRATION ON PH2: NO
MODELS MODIFIED: NO

Scores are model-predicted melanoma probability estimates, not calibrated clinical probabilities.
This is not clinical validation. PH2 was used previously in this project.

## Cohort

- 40 melanoma, 160 nevi (80 common + 80 atypical), 200 total
- Same cases for both models
- Ensemble 0.70/0.30, threshold 0.50 (frozen; not searched on PH2)

## Frozen 83.42% baseline

{
  "threshold": 0.5,
  "accuracy": 0.33,
  "precision": 0.22988505747126436,
  "recall": 1.0,
  "sensitivity": 1.0,
  "specificity": 0.1625,
  "f1": 0.37383177570093457,
  "auc": 0.83546875,
  "pr_auc": 0.6592606644477185,
  "tp": 40,
  "fp": 134,
  "tn": 26,
  "fn": 0,
  "n_positive": 40,
  "n_negative": 160
}

CM TP=40 TN=26 FP=134 FN=0

## Mild-contrast augmentation

{
  "threshold": 0.5,
  "accuracy": 0.285,
  "precision": 0.2154696132596685,
  "recall": 0.975,
  "sensitivity": 0.975,
  "specificity": 0.1125,
  "f1": 0.35294117647058826,
  "auc": 0.8484375,
  "pr_auc": 0.6865249076730708,
  "tp": 39,
  "fp": 142,
  "tn": 18,
  "fn": 1,
  "n_positive": 40,
  "n_negative": 160
}

CM TP=39 TN=18 FP=142 FN=1

## Comparison (same 200 PH2 cases)

| Metric | Frozen 83.42% baseline | Mild-contrast augmentation |
|---|---:|---:|
| Accuracy | 0.3300 | 0.2850 |
| Sensitivity / recall | 1.0000 | 0.9750 |
| Specificity | 0.1625 | 0.1125 |
| Precision | 0.2299 | 0.2155 |
| F1 | 0.3738 | 0.3529 |
| ROC-AUC | 0.8355 | 0.8484 |
| PR-AUC | 0.6593 | 0.6865 |
| TP | 40 | 39 |
| TN | 26 | 18 |
| FP | 134 | 142 |
| FN | 0 | 1 |

Frozen baseline sensitivity 1.000 vs mild-contrast 0.975; specificity 0.163 vs 0.113; precision 0.230 vs 0.215; accuracy 0.330 vs 0.285. False negatives: baseline 0/40, mild-contrast 1/40. False positives: baseline 134/160, mild-contrast 142/160. Small PH2 melanoma n=40; treat differences as descriptive, not a ranking. PH2 was not used to select or replace the frozen baseline.

Not clinically validated. No model was modified from these PH2 results.

