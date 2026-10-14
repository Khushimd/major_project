# Frozen baseline threshold-optimization experiment

Research / decision-support evaluation only. **Not clinically validated.**
Scores are AI probability estimates, not calibrated clinical probabilities.
This experiment does not diagnose benignity and does not replace the 83.42% baseline.

## 1. Objective

Ask whether a different decision threshold on the existing 0.70/0.30 ensemble
can reduce false negatives without an unacceptable drop in specificity, precision,
F1, or accuracy. Models were not retrained. TTA was not used.

## 2. Frozen setup (unchanged)

- Dataset: `dataset/processed/final_melanoma_benign/`
- Validation 2438 (1043 melanoma / 1395 benign); test 2437 (1041 / 1396)
- EfficientNetB0 0.70 + MobileNetV2 0.30 soft voting
- Current operating threshold: 0.50
- Scores reused from frozen baseline inference artifacts (no weight reload)

## 3. Protocol

1. Store validation and test ensemble scores.
2. Search thresholds 0.20-0.70 step 0.01 on **validation only**.
3. Apply preferred constraints, then lock one threshold.
4. Evaluate the frozen test set **once**.
5. No post-test retuning.

## 4. Validation metrics at current 0.50

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

Validation ROC-AUC 0.9118; PR-AUC 0.8893 
(unchanged across thresholds).

## 5. Constrained selection

Feasible validation thresholds: 6

Among validation thresholds meeting specificity>=0.82, precision>=0.78, F1>=0.80, accuracy>=0.82: highest sensitivity, then lowest FN, then highest F1, then highest specificity, then highest precision.

**Selected validation threshold: 0.50**

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

## 6. Highest-sensitivity validation threshold (not selected unless it met constraints)

{
  "threshold": 0.2,
  "accuracy": 0.6776045939294504,
  "precision": 0.5719082260772244,
  "recall": 0.9798657718120806,
  "sensitivity": 0.9798657718120806,
  "specificity": 0.45161290322580644,
  "f1": 0.72226148409894,
  "auc": 0.911757853173744,
  "pr_auc": 0.8893139033924969,
  "tp": 1022,
  "fp": 765,
  "tn": 630,
  "fn": 21,
  "n_positive": 1043,
  "n_negative": 1395
}

This row is reported for trade-off context only.

## 7. Validation thresholds with FN ↓ vs 0.50 and spec/prec still reasonably high

None in the 0.20-0.70 grid.

## 8. Full validation trade-off table

| Threshold | Acc | Sens | Spec | Prec | F1 | TP | TN | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.6776 | 0.9799 | 0.4516 | 0.5719 | 0.7223 | 1022 | 630 | 765 | 21 |  ← max sensitivity
| 0.21 | 0.6969 | 0.9760 | 0.4882 | 0.5878 | 0.7337 | 1018 | 681 | 714 | 25 |
| 0.22 | 0.7141 | 0.9722 | 0.5211 | 0.6029 | 0.7442 | 1014 | 727 | 668 | 29 |
| 0.23 | 0.7272 | 0.9674 | 0.5477 | 0.6152 | 0.7521 | 1009 | 764 | 631 | 34 |
| 0.24 | 0.7363 | 0.9636 | 0.5663 | 0.6242 | 0.7576 | 1005 | 790 | 605 | 38 |
| 0.25 | 0.7453 | 0.9607 | 0.5842 | 0.6334 | 0.7634 | 1002 | 815 | 580 | 41 |
| 0.26 | 0.7531 | 0.9578 | 0.6000 | 0.6416 | 0.7685 | 999 | 837 | 558 | 44 |
| 0.27 | 0.7580 | 0.9530 | 0.6122 | 0.6476 | 0.7711 | 994 | 854 | 541 | 49 |
| 0.28 | 0.7654 | 0.9501 | 0.6272 | 0.6559 | 0.7760 | 991 | 875 | 520 | 52 |
| 0.29 | 0.7687 | 0.9434 | 0.6380 | 0.6608 | 0.7773 | 984 | 890 | 505 | 59 |
| 0.30 | 0.7748 | 0.9415 | 0.6502 | 0.6680 | 0.7815 | 982 | 907 | 488 | 61 |
| 0.31 | 0.7769 | 0.9367 | 0.6573 | 0.6715 | 0.7822 | 977 | 917 | 478 | 66 |
| 0.32 | 0.7801 | 0.9291 | 0.6688 | 0.6771 | 0.7833 | 969 | 933 | 462 | 74 |
| 0.33 | 0.7830 | 0.9243 | 0.6774 | 0.6818 | 0.7847 | 964 | 945 | 450 | 79 |
| 0.34 | 0.7888 | 0.9233 | 0.6882 | 0.6888 | 0.7890 | 963 | 960 | 435 | 80 |
| 0.35 | 0.7941 | 0.9185 | 0.7011 | 0.6967 | 0.7924 | 958 | 978 | 417 | 85 |
| 0.36 | 0.7986 | 0.9147 | 0.7118 | 0.7035 | 0.7953 | 954 | 993 | 402 | 89 |
| 0.37 | 0.8023 | 0.9089 | 0.7226 | 0.7101 | 0.7973 | 948 | 1008 | 387 | 95 |
| 0.38 | 0.8052 | 0.9032 | 0.7319 | 0.7158 | 0.7986 | 942 | 1021 | 374 | 101 |
| 0.39 | 0.8080 | 0.8955 | 0.7427 | 0.7224 | 0.7997 | 934 | 1036 | 359 | 109 |
| 0.40 | 0.8093 | 0.8859 | 0.7520 | 0.7276 | 0.7990 | 924 | 1049 | 346 | 119 |
| 0.41 | 0.8109 | 0.8811 | 0.7584 | 0.7317 | 0.7995 | 919 | 1058 | 337 | 124 |
| 0.42 | 0.8117 | 0.8734 | 0.7656 | 0.7359 | 0.7988 | 911 | 1068 | 327 | 132 |
| 0.43 | 0.8167 | 0.8696 | 0.7771 | 0.7447 | 0.8023 | 907 | 1084 | 311 | 136 |
| 0.44 | 0.8203 | 0.8639 | 0.7878 | 0.7527 | 0.8045 | 901 | 1099 | 296 | 142 |
| 0.45 | 0.8191 | 0.8543 | 0.7928 | 0.7551 | 0.8016 | 891 | 1106 | 289 | 152 |
| 0.46 | 0.8199 | 0.8466 | 0.8000 | 0.7599 | 0.8009 | 883 | 1116 | 279 | 160 |
| 0.47 | 0.8199 | 0.8380 | 0.8065 | 0.7640 | 0.7993 | 874 | 1125 | 270 | 169 |
| 0.48 | 0.8216 | 0.8332 | 0.8129 | 0.7690 | 0.7998 | 869 | 1134 | 261 | 174 |
| 0.49 | 0.8236 | 0.8274 | 0.8208 | 0.7754 | 0.8006 | 863 | 1145 | 250 | 180 |
| 0.50 | 0.8265 | 0.8226 | 0.8294 | 0.7828 | 0.8022 | 858 | 1157 | 238 | 185 |  ← current 0.50  ← selected
| 0.51 | 0.8294 | 0.8169 | 0.8387 | 0.7911 | 0.8038 | 852 | 1170 | 225 | 191 |
| 0.52 | 0.8302 | 0.8111 | 0.8444 | 0.7959 | 0.8034 | 846 | 1178 | 217 | 197 |
| 0.53 | 0.8269 | 0.8006 | 0.8466 | 0.7960 | 0.7983 | 835 | 1181 | 214 | 208 |
| 0.54 | 0.8302 | 0.7958 | 0.8559 | 0.8050 | 0.8004 | 830 | 1194 | 201 | 213 |
| 0.55 | 0.8294 | 0.7881 | 0.8602 | 0.8083 | 0.7981 | 822 | 1200 | 195 | 221 |
| 0.56 | 0.8326 | 0.7852 | 0.8681 | 0.8166 | 0.8006 | 819 | 1211 | 184 | 224 |
| 0.57 | 0.8351 | 0.7795 | 0.8767 | 0.8254 | 0.8018 | 813 | 1223 | 172 | 230 |
| 0.58 | 0.8343 | 0.7728 | 0.8803 | 0.8284 | 0.7996 | 806 | 1228 | 167 | 237 |
| 0.59 | 0.8335 | 0.7651 | 0.8846 | 0.8321 | 0.7972 | 798 | 1234 | 161 | 245 |
| 0.60 | 0.8318 | 0.7565 | 0.8882 | 0.8349 | 0.7938 | 789 | 1239 | 156 | 254 |
| 0.61 | 0.8322 | 0.7488 | 0.8946 | 0.8416 | 0.7925 | 781 | 1248 | 147 | 262 |
| 0.62 | 0.8306 | 0.7373 | 0.9004 | 0.8469 | 0.7883 | 769 | 1256 | 139 | 274 |
| 0.63 | 0.8277 | 0.7267 | 0.9032 | 0.8488 | 0.7831 | 758 | 1260 | 135 | 285 |
| 0.64 | 0.8249 | 0.7181 | 0.9047 | 0.8492 | 0.7782 | 749 | 1262 | 133 | 294 |
| 0.65 | 0.8220 | 0.7085 | 0.9068 | 0.8504 | 0.7730 | 739 | 1265 | 130 | 304 |
| 0.66 | 0.8236 | 0.7047 | 0.9125 | 0.8576 | 0.7737 | 735 | 1273 | 122 | 308 |
| 0.67 | 0.8224 | 0.6980 | 0.9154 | 0.8605 | 0.7708 | 728 | 1277 | 118 | 315 |
| 0.68 | 0.8216 | 0.6913 | 0.9190 | 0.8645 | 0.7682 | 721 | 1282 | 113 | 322 |
| 0.69 | 0.8191 | 0.6826 | 0.9211 | 0.8662 | 0.7635 | 712 | 1285 | 110 | 331 |
| 0.70 | 0.8191 | 0.6788 | 0.9240 | 0.8698 | 0.7625 | 708 | 1289 | 106 | 335 |

## 9. Frozen test evaluation

One evaluation at locked threshold 0.50.

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

Test confusion matrix: TP 848 / TN 1185 / FP 211 / FN 193

ROC-AUC 0.9127; PR-AUC 0.8885 
(same ranking metrics as the frozen 0.50 baseline, as expected).

## 10. Baseline vs optimized threshold (frozen test)

| Metric | Baseline 0.50 | Optimized | Change |
|---|---:|---:|---:|
| Accuracy | 83.42% | 83.42% | +0.00 pp |
| Sensitivity | 81.46% | 81.46% | +0.00 pp |
| Specificity | 84.89% | 84.89% | +0.00 pp |
| Precision | 80.08% | 80.08% | +0.00 pp |
| F1 | 0.8076 | 0.8076 | +0.0000 |
| ROC-AUC | 0.9127 | 0.9127 | +0.0000 |
| PR-AUC | 0.8885 | 0.8885 | +0.0000 |
| TP | 848 | 848 | +0 |
| TN | 1185 | 1185 | +0 |
| FP | 211 | 211 | +0 |
| FN | 193 | 193 | +0 |

FN reduction = 0

## 11. Interpretation

**threshold optimization alone did not provide a satisfactory FN reduction**

The constrained validation threshold did not reduce frozen-test false negatives in a meaningful way versus 0.50.

Better than 0.50 as a balanced operating point: False
Replace frozen baseline: False

## 12. Optional three-zone interface note

{
  "supported_as_interface_idea_only": true,
  "not_clinical_thresholds": true,
  "cannot_rule_out_melanoma": true,
  "validation_benign": {
    "n": 1395,
    "mean": 0.3040577791011675,
    "median": 0.2125395089387893,
    "min": 0.0538874924182891,
    "max": 0.9962264418601988,
    "p10": 0.1257050311565399,
    "p25": 0.1576525695621967,
    "p75": 0.398263394087553,
    "p90": 0.6178587567806244
  },
  "validation_melanoma": {
    "n": 1043,
    "mean": 0.7800078536975816,
    "median": 0.8931676030158997,
    "min": 0.1098253719508647,
    "max": 0.9999374330043792,
    "p10": 0.3840850535035133,
    "p25": 0.6065896853804588,
    "p75": 0.9893742054700851,
    "p90": 0.9961343514919279
  },
  "overlap_benign_p90_vs_melanoma_p10": true,
  "possible_interface_bands_validation_only": {
    "lower_suspicion": "ensemble_prob < 0.35",
    "intermediate_clinical_review": "0.35 <= ensemble_prob < 0.55",
    "higher_suspicion": "ensemble_prob >= 0.55",
    "disclaimer": "Descriptive score bands from validation overlap only. Not clinical cutoffs. Intermediate cases still require clinician review. A low score does not rule out melanoma."
  }
}

## 13. What was not modified

- `models/saved/`
- `models/experiments/final_melanoma_benign_baseline/`
- `models/experiments/final_melanoma_benign_class_weighted/`
- frozen CSVs, production inference config, dashboard

Script: `scripts/eval_baseline_threshold_optimization.py`.

