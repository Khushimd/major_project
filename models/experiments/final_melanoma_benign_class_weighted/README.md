# final_melanoma_benign_class_weighted

One controlled experiment: same frozen `final_melanoma_benign` dataset and the same
83.4% baseline settings, with balanced class weights from the training split only.

- benign_weight = 0.8522318754
- melanoma_weight = 1.2097597278

Start here:
- `CLASS_WEIGHTED_EXPERIMENT_REPORT.md`
- `comparison_report.md`
- `training_summary.json`
- `class_weights.json`

This directory is the only write target. Frozen baseline, production, Exp2-C,
Derm7pt external, clean DICM, and dataset CSVs were not modified.

Not clinically validated. `malignant_prob` is a model score, not a calibrated clinical probability.
