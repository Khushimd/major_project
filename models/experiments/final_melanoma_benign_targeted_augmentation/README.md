# Targeted low-contrast augmentation experiment

Research / decision-support only. **Not clinically validated.**

## Objective

Test whether extra conservative train-time photometric jitter (brightness, contrast,
mild gamma, mild saturation) reduces false negatives versus the frozen 83.42% baseline
without breaking specificity, precision, F1, or accuracy constraints.

## Frozen baseline

Ensemble 0.70/0.30, threshold 0.50. Test acc 83.42%, sens 81.46%, spec 84.89%,
prec 80.08%, F1 0.8076, ROC-AUC 0.9127, PR-AUC 0.8885, FN 193.

## FN-analysis motivation

Validation-only analysis: FN images were flatter/brighter than TP melanoma
(contrast ~25 vs ~48; brightness ~155 vs ~137). Most FNs scored 0.30–0.50.
Validation FNs were not oversampled.

## Exact extra augmentation (train only, on top of baseline aug)

{
  "keeps_baseline_augmentation": true,
  "extra_brightness": {
    "factor": [
      0.85,
      1.15
    ],
    "probability": 0.3
  },
  "extra_contrast": {
    "factor": [
      0.75,
      1.25
    ],
    "probability": 0.4
  },
  "extra_gamma": {
    "factor": [
      0.85,
      1.15
    ],
    "probability": 0.2
  },
  "extra_saturation": {
    "factor": [
      0.9,
      1.1
    ],
    "probability": 0.15
  },
  "applied_to": "train only",
  "validation_fn_oversampled": false
}

## Training configuration

Same 3-stage fine-tune, BCE + label smoothing 0.05, seed 42, no class weights,
no oversampling. Ensemble and threshold locked at 0.70/0.30 and 0.50.

## Validation results

{
  "threshold": 0.5,
  "accuracy": 0.8125512715340443,
  "precision": 0.7547826086956522,
  "recall": 0.8322147651006712,
  "sensitivity": 0.8322147651006712,
  "specificity": 0.7978494623655914,
  "f1": 0.791609667122663,
  "auc": 0.9043412818688852,
  "pr_auc": 0.8837668368893384,
  "tp": 868,
  "fp": 282,
  "tn": 1113,
  "fn": 175,
  "n_positive": 1043,
  "n_negative": 1395
}

Constraints satisfied: False
Meaningful FN/sensitivity gain vs baseline val: True
Test evaluation performed: False

## Test results

{
  "evaluated_once": false,
  "post_test_tuning": false,
  "threshold": 0.5,
  "ensemble_weights": {
    "efficientnet": 0.7,
    "mobilenet": 0.3
  },
  "metrics": null,
  "reason": "Validation did not both meet constraints and show a meaningful FN/sensitivity gain. Test not evaluated."
}

## Comparison and recommendation

Targeted augmentation did not provide a sufficiently balanced validation improvement (failures: ['specificity 0.7978 < 0.82', 'precision 0.7548 < 0.78', 'f1 0.7916 < 0.80', 'accuracy 0.8126 < 0.82']). Retain the 83.42% baseline.

Replace frozen baseline: False

NO test images were used for model selection. Nothing else was trained after this run.

