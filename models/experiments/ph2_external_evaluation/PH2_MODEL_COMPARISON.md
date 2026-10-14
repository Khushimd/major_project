# PH2 EXTERNAL EVALUATION — MODEL COMPARISON

Analysis only. No retraining, tuning, calibration, or file overwrites of models or datasets.
Source: saved files in `models/experiments/ph2_external_evaluation/`.
Scores below are **model-predicted melanoma scores / probability estimates**, not calibrated clinical probabilities.
This is **not** clinical validation.

PH2 was used previously in this project. This is an external evaluation on a **fixed 200-case PH2 cohort**, not a never-before-seen dataset.

The historical internal ~92.28% result is **not** a PH2 result and is **not** a pure melanoma-vs-benign number (original DICM `class=1` included other malignant diagnoses). It is quoted only as historical internal reference.

## Cohort and protocol (from saved reports)

| Item | Saved value |
|---|---|
| Cases | 200 |
| Melanoma | 40 |
| Strict benign nevi | 160 (80 common + 80 atypical) |
| Same cases for both models | yes (`ph2_predictions.csv`, 200 rows, both score columns present) |
| PH2 threshold tuning | **NO** |
| PH2 ensemble-weight tuning | **NO** |
| PH2 calibration / training | **NO** |
| Old model weights / threshold | EfficientNet 0.75 / MobileNet 0.25, threshold **0.30** (frozen Exp2-C) |
| Class-weighted weights / threshold | EfficientNet 0.50 / MobileNet 0.50, threshold **0.45** (validation-selected) |

## PH2 metrics

| Metric | Old frozen 92% model | Class-weighted model | Difference (CW − old) | Higher on PH2 |
|---|---:|---:|---:|---|
| Accuracy | 0.2050 | 0.2650 | +0.0600 | class-weighted |
| Sensitivity / Recall | 1.0000 | 1.0000 | 0.0000 | tie |
| Specificity | 0.0063 | 0.0813 | +0.0750 | class-weighted |
| Precision | 0.2010 | 0.2139 | +0.0129 | class-weighted |
| F1-score | 0.3347 | 0.3524 | +0.0177 | class-weighted |
| ROC-AUC | 0.6573 | 0.8344 | +0.1770 | class-weighted |
| PR-AUC | 0.4308 | 0.6304 | +0.1996 | class-weighted |
| TP | 40 | 40 | 0 | tie |
| TN | 1 | 13 | +12 | class-weighted |
| FP | 159 | 147 | −12 | class-weighted (fewer FP) |
| FN | 0 | 0 | 0 | tie |
| Predicted melanoma | 199 | 187 | −12 | — |
| Predicted benign | 1 | 13 | +12 | — |

Confusion matrices (saved):

- Old: TP 40 / TN 1 / FP 159 / FN 0
- Class-weighted: TP 40 / TN 13 / FP 147 / FN 0

No overall “best model” ranking is assigned. The table above is per-metric only.

## Prediction-score distribution (saved)

Model-predicted melanoma scores, not clinical probabilities.

| Group | Old mean | Old median | Old min–max | CW mean | CW median | CW min–max |
|---|---:|---:|---|---:|---:|---|
| All (n=200) | 0.724 | 0.700 | 0.295–1.000 | 0.749 | 0.772 | 0.310–0.994 |
| Melanoma (n=40) | 0.805 | 0.921 | 0.339–1.000 | 0.889 | 0.937 | 0.456–0.994 |
| Strict benign (n=160) | 0.704 | 0.673 | 0.295–1.000 | 0.714 | 0.738 | 0.310–0.993 |
| Common nevus (n=80) | 0.681 | 0.646 | 0.308–1.000 | 0.696 | 0.722 | 0.316–0.993 |
| Atypical nevus (n=80) | 0.727 | 0.702 | 0.295–1.000 | 0.731 | 0.768 | 0.310–0.982 |

Subtype errors (saved `subtype_results.csv`):

- Old: all 80 common nevi predicted melanoma; 79/80 atypical nevi predicted melanoma; 40/40 melanoma predicted melanoma.
- Class-weighted: 74/80 common and 73/80 atypical predicted melanoma; 40/40 melanoma predicted melanoma.

## Answers to the review questions

### 1. Which model performed better on PH2 for each metric?

See the difference column above. Class-weighted is numerically higher on accuracy, specificity, precision, F1, ROC-AUC, PR-AUC, and TN, and lower on FP. Sensitivity, TP, and FN are identical.

### 2. Did class weighting improve external performance versus old/new baseline?

**Versus the old frozen 92% model, on this PH2 file:** several PH2 metrics are higher for the class-weighted model, especially ROC-AUC (+0.177) and PR-AUC (+0.200). Thresholded accuracy/specificity remain low for both.

**Versus the new 83.4% unweighted baseline, on PH2:** the saved PH2 evaluation **did not score** the 83.4% baseline on PH2. That comparison is not in these files and is not estimated here.

**Internal (not PH2), from the class-weighted experiment summary copied into `ph2_external_results.json`:** class-weighted frozen test vs 83.4% baseline was already recorded as a **trade-off, not a clear improvement** (sensitivity +0.0365, specificity −0.0394, F1 +0.0004, ROC-AUC −0.0005). That is an internal-test statement, not a PH2 statement.

### 3. Strengths and weaknesses on PH2

**Old frozen 92% model**

- Strength: no PH2 melanoma missed (FN=0, sensitivity=1.0).
- Weakness: almost no negative class (TN=1, specificity=0.0063). Predicted melanoma on 199/200 cases. ROC-AUC 0.6573 shows weak ranking of melanoma vs nevus. Benign median score 0.673 is far above the frozen 0.30 cut.

**Class-weighted model**

- Strength: also FN=0; better ranking (ROC-AUC 0.8344, PR-AUC 0.6304); melanoma scores sit higher (median 0.937 vs 0.921) with a higher melanoma minimum (0.456 vs 0.339).
- Weakness: still predicts melanoma on 187/200 cases (specificity 0.0813). Benign scores remain high (median 0.738 vs threshold 0.45). Specificity is improved only from 1/160 to 13/160 true negatives.

### 4–6. Is additional training justified?

**Additional training is not justified by these PH2 results. Freeze the class-weighted model as the completed experiment artifact.**

Reasons from the saved numbers only:

- Both models already detect all 40 PH2 melanomas. The PH2 failure is false positives at the **already frozen** operating points, not missed melanoma.
- Class-weighting was the controlled next training experiment. Its own internal test was a trade-off, not a clear improvement.
- PH2 ROC-AUC 0.834 vs internal class-weighted ROC-AUC 0.912 is a **domain-shift / generalization gap**, not evidence that more epochs on the same development set would fix PH2 specificity.
- PH2 must not be used to retune threshold, weights, or training. Using PH2 to decide another training loop would leak this evaluation set.

What these files do **not** support: starting another training experiment, searching a PH2 threshold, or claiming the class-weighted model is ready for clinical use.

### 7. Same 200 cases? Any PH2 tuning?

Yes, same 200 cases: `ph2_predictions.csv` has 200 rows with both models’ ensemble scores and predictions. Reports state `THRESHOLD TUNING ON PH2: NO`, `ENSEMBLE-WEIGHT TUNING ON PH2: NO`, `ph2_used_for_tuning: false`. Old weights/threshold are the frozen Exp2-C 0.75/0.25 and 0.30. Class-weighted 0.50/0.50 and 0.45 were read from the class-weighted experiment’s validation-selected config.

### 8. Warnings, inconsistencies, limitations

- **Different frozen thresholds** (0.30 vs 0.45) mean thresholded metrics are not a pure like-for-like training comparison; ranking metrics (ROC-AUC, PR-AUC) do not depend on those cuts.
- **Historical 92.28% must not be compared as a PH2 or pure melanoma-vs-benign result.**
- **83.4% baseline was not evaluated on PH2** in this folder.
- PH2 was previously used in the project.
- n=40 melanoma: binomial variance is large; treat small count differences (TN 1 vs 13) as descriptive.
- Benign max scores remain ~0.99 for both models: classes are not well separated at the frozen cuts.
- Old model min overall score is 0.295, just under 0.30, which is why TN=1 rather than 0.
- Protected-file snapshots in this folder record that production, Exp2-C, baseline, class-weighted, and frozen CSVs were not modified by the PH2 run.
- Not clinically validated; scores are not calibrated clinical probabilities.

## Key findings

1. On the fixed PH2 cohort, both frozen cutoffs call most nevi melanoma. Sensitivity is 1.0 for both; specificity is 0.006 (old) and 0.081 (class-weighted).
2. Ranking is numerically stronger for the class-weighted model (ROC-AUC 0.834 vs 0.657; PR-AUC 0.630 vs 0.431).
3. This does not revive the historical internal 92.28% as an external melanoma-vs-benign result.

## Class-weighting effect

On PH2 versus the old 92% model: higher ranking metrics and a small specificity gain (1 → 13 true negatives), with sensitivity unchanged at 1.0. On the internal 83.4% baseline comparison (not PH2): previously saved as a trade-off, not a clear improvement. The 83.4% model was not scored on PH2.

## Whether more training is justified

**No. Freeze the class-weighted model.** These PH2 results do not justify another training experiment.

## Evaluation limitations

PH2 is small, previously used, and domain-shifted relative to the development set. Frozen thresholds differ between models. The 83.4% baseline has no PH2 numbers here. Results are not clinical validation.
