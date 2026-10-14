# Mild contrast-only augmentation experiment

Research / decision-support only. **Not clinically validated.**

## Objective

Test whether ONE extra mild train-time contrast jitter can raise accuracy,
recall, and precision together versus the frozen 83.42% baseline, without
breaking specificity / F1 floors. This is a different protocol from the
failed stacked photometric experiment.

## Frozen baseline

Ensemble 0.70/0.30, threshold 0.50. Test acc 83.42%, sens 81.46%, spec 84.89%,
prec 80.08%, F1 0.8076, ROC-AUC 0.9127, PR-AUC 0.8885, FN 193.

## Why this is milder

Stacked extra brightness/contrast/gamma/saturation dropped val spec/prec/acc.
FN analysis showed missed melanomas were flatter (contrast ~25 vs ~48).
This run adds only extra contrast 0.85–1.15 with p=0.25. Validation FNs
were not oversampled.

## Exact extra augmentation (train only, on top of baseline aug)

{
  "keeps_baseline_augmentation": true,
  "extra_brightness": null,
  "extra_gamma": null,
  "extra_saturation": null,
  "extra_contrast": {
    "factor": [
      0.85,
      1.15
    ],
    "probability": 0.25
  },
  "applied_to": "train only",
  "validation_fn_oversampled": false,
  "rationale": "Stacked photometric jitter already failed constraints. FN analysis showed missed melanomas were flatter than true positives, so this run adds only mild extra contrast."
}

## Training configuration

Same 3-stage fine-tune, BCE + label smoothing 0.05, seed 42, no class weights,
no oversampling. Ensemble and threshold locked at 0.70/0.30 and 0.50.

## Validation results

{
  "threshold": 0.5,
  "accuracy": 0.8150123051681706,
  "precision": 0.7578397212543554,
  "recall": 0.8341323106423778,
  "sensitivity": 0.8341323106423778,
  "specificity": 0.8007168458781362,
  "f1": 0.7941579187585578,
  "auc": 0.9075481877820046,
  "pr_auc": 0.8846113966348468,
  "tp": 870,
  "fp": 278,
  "tn": 1117,
  "fn": 173,
  "n_positive": 1043,
  "n_negative": 1395
}

Constraints satisfied: False
Balanced headline gain (acc/recall/prec all hold, at least one better): False
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
  "reason": "Validation did not both meet constraints and improve accuracy/recall/precision without hurting any of those three. Test not evaluated."
}

## Comparison and recommendation

Mild contrast augmentation did not provide a balanced validation improvement (failures: ['specificity 0.8007 < 0.82', 'precision 0.7578 < 0.78', 'f1 0.7942 < 0.80', 'accuracy 0.8150 < 0.82']). Retain the 83.42% baseline.

Replace frozen baseline: False

NO test images were used for model selection. Nothing else was trained after this run.

