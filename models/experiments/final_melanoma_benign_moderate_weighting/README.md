# Moderate melanoma weighting experiment

Research / decision-support evaluation only. **Not clinically validated.**
Scores are AI probability estimates. This does not replace the frozen 83.42% baseline
unless the report below explicitly recommends it (it does not do so automatically).

## Objective

Ask whether a *small*, predefined melanoma class-weight can reduce false negatives
versus the frozen 83.42% baseline while keeping specificity, precision, F1, and
accuracy in an acceptable band. This is not the previous balanced-weight experiment.

## Dataset

Frozen `dataset/processed/final_melanoma_benign/`.
Train 11379 (4703 melanoma / 6676 benign). Val 2438 (1043 / 1395). Test 2437 (1041 / 1396).
Splits, preprocessing cache, and labels were not modified.

## Weighting configurations

benign_weight : melanoma_weight, applied only during `model.fit` via `class_weight`.

- `W_1.00`: 1.00 : 1.00 (control/reference)
- `W_1.05`: 1.00 : 1.05 (moderate candidate)
- `W_1.10`: 1.00 : 1.10 (moderate candidate)
- `W_1.15`: 1.00 : 1.15 (moderate candidate)

No oversampling, undersampling, HAM10000, PH2, or Derm7pt.

## Fixed operating point

Ensemble EfficientNet 0.70 / MobileNet 0.30. Decision threshold 0.50.
No ensemble-weight search. No threshold search.

## Validation selection rule

Compare the four configurations on validation at 0.50.
Constraints: specificity >= 82%, precision >= 78%, F1 >= 0.80, accuracy >= 82%.
Among feasible: highest sensitivity, then lowest FN, then highest F1, then
highest specificity, then highest precision, then highest accuracy.
Constraints were not relaxed.

{
  "feasible": false,
  "n_feasible": 0,
  "selected_id": "retained_baseline",
  "selected": null,
  "rule": "No moderate-weighting configuration met specificity>=0.82, precision>=0.78, F1>=0.80, and accuracy>=0.82 on validation at threshold 0.50. Constraints were not relaxed. Retain the frozen 83.42% baseline."
}

## Frozen test results

{
  "evaluated_once": false,
  "post_test_tuning": false,
  "threshold": 0.5,
  "ensemble_weights": {
    "efficientnet": 0.7,
    "mobilenet": 0.3
  },
  "selected_id": "retained_baseline",
  "metrics": null,
  "reason": "No constrained validation winner; test not used to pick a weight. Baseline retained."
}

## Comparison with baseline and previous ablations

Candidate rows 1.00–1.15 in `comparison_table.csv` are **validation**.
Baseline, threshold optimization, and balanced class-weighting rows are **frozen test**.
The selected-model row is the single frozen-test evaluation, or the retained baseline.

Previous balanced weights: benign=0.852232, melanoma=1.209760.

## Did FN decrease?

No new configuration was selected. Frozen-test FN remains 193 at the 83.42% baseline.

## Should this replace the baseline?

False

Moderate weighting did not provide a sufficiently favorable operating trade-off under the stated constraints. Retain the 83.42% baseline.

## What was not modified

- `models/saved/`
- `models/experiments/final_melanoma_benign_baseline/`
- `models/experiments/final_melanoma_benign_class_weighted/`
- Flask app and doctor dashboard

Script: `scripts/train_final_melanoma_benign_moderate_weighting.py`.

