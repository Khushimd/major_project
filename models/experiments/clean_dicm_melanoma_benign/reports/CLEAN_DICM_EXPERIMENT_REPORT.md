# Clean DICM Melanoma-vs-Benign Experiment

This is an internal research experiment. It is **not clinically validated**.
Reported scores are **AI probability estimates / model scores**, not clinically calibrated probabilities.
This report does not claim state-of-the-art performance and does not compare against published papers.

## 1. Dataset definition

Source: `dataset/processed/dicm_clean_melanoma_benign/`.
Melanoma (label=1): official DICM diagnosis == melanoma AND benign_malignant in {malignant, indeterminate/malignant}.
Benign (label=0): diagnosis in the pre-specified clean benign list AND benign_malignant == benign.
Official DICM class was not used. ISIC 2020 and HAM10000 were not added. Images were not deleted to rebalance.

## 2. Dataset counts

- Total: 11373 (melanoma 6787, benign 4586)
- Train: {'total': 7961, 'melanoma': 4747, 'benign': 3214}
- Validation: {'total': 1706, 'melanoma': 1016, 'benign': 690}
- Test: {'total': 1706, 'melanoma': 1024, 'benign': 682}

## 3. Leakage checks

Non-empty patient IDs, lesion IDs, and image IDs have zero overlap among the new train, validation, and test splits.
Details: `{'image_id': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'lesion_id': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'patient_id': {'train_val': 0, 'train_test': 0, 'val_test': 0}}`.

## 4. Preprocessing

Read image, BGR to RGB, downscale to max side 1024 when larger (many DICM images are ~30MP,
which makes full-resolution DullRazor intractable on CPU), then DullRazor-style hair removal,
light denoising, resize 224x224, scale to [0,1].
ImageNet preprocessing is applied inside EfficientNetB0 and MobileNetV2.
CLAHE and aggressive lesion cropping were not used.
Cached copies were written only under `C:\Projects\major-project\models\experiments\clean_dicm_melanoma_benign\preprocessed`.

## 5. Augmentation

Training only, via `get_tf_data_augmentation_fn()`: horizontal/vertical flip, small rotation, small zoom,
small translation, moderate brightness/contrast, mild saturation/hue, light Gaussian noise, mild blur,
and occasional synthetic hair. No extra augmentation was added.

## 6. Model architecture

Two independent ImageNet-initialized CNNs with the existing shared classification head:
EfficientNetB0 and MobileNetV2, sigmoid melanoma score output.
New files only; production weights were not loaded or overwritten.

## 7. Training configuration

- Seed: 42
- Optimizer: AdamW, 3-stage schedule (head 1e-3 / partial 1e-4 / final 1e-5)
- Loss: BinaryCrossentropy(label_smoothing=0.05)
- Experiment A class weights: none
- Experiment B class weights from train split only: {0: 1.2384878655880522, 1: 0.8385295976406151}
- Checkpoint / early stopping monitor: validation PR-AUC
- Test labels were not used for loss, weights, ensemble, or threshold

## 8. Experiment A results (validation)

- Ensemble weights: EfficientNet 0.60 / MobileNet 0.40
- Threshold: 0.50 (sens>=0.80 and spec>=0.65, max F1)
- Accuracy 0.7784; sensitivity 0.8563; specificity 0.6638
- Precision 0.7895; F1 0.8215; ROC-AUC 0.8707; PR-AUC 0.9113
- EfficientNet val ROC-AUC 0.8656
- MobileNet val ROC-AUC 0.8466

## 9. Experiment B results (validation)

- Ensemble weights: EfficientNet 0.50 / MobileNet 0.50
- Threshold: 0.45 (sens>=0.80 and spec>=0.65, max F1)
- Accuracy 0.7691; sensitivity 0.8396; specificity 0.6652
- Precision 0.7869; F1 0.8124; ROC-AUC 0.8650; PR-AUC 0.9064
- EfficientNet val ROC-AUC 0.8564
- MobileNet val ROC-AUC 0.8479

## 10. Validation-based selection

Selected experiment: `experiment_A_bce_smoothing`
Rule: validation ensemble ROC-AUC, then PR-AUC, then F1 at the validation-selected threshold
Frozen ensemble weights: EfficientNet 0.60 / MobileNet 0.40
Frozen threshold: 0.50
Threshold rule: sens>=0.80 and spec>=0.65, max F1
The old 0.75/0.25 weights and 0.30 threshold were not reused automatically.
The test set was not used for this decision.

## 11. Frozen test results

One evaluation after architecture, training, checkpoints, ensemble weights, and threshold were frozen.
- Accuracy: 0.7708 (95% bootstrap CI 0.7515-0.7913)
- Sensitivity / recall: 0.8311 (0.8083-0.8532)
- Specificity: 0.6804 (0.6462-0.7145)
- Precision: 0.7961 (0.7725-0.8204)
- F1-score: 0.8132 (0.7950-0.8320)
- ROC-AUC: 0.8521 (0.8350-0.8703)
- PR-AUC: 0.9025 (0.8871-0.9177)

## 12. Confusion matrix

- TP 851
- TN 464
- FP 218
- FN 173

## 13. ROC-AUC and PR-AUC

Frozen test ROC-AUC 0.8521; PR-AUC 0.9025.
Curve data: `models/experiments/clean_dicm_melanoma_benign/evaluation/test_roc_curve.json` and `test_pr_curve.json`.

## 14. Limitations

- Single-source DICM images; hospital / camera confounding may remain.
- Missing patient IDs required lesion- or image-level grouping for some rows.
- No external validation in this run.
- Scores are not claimed to be clinically calibrated probabilities.
- Grad-CAM is qualitative only and does not prove correct lesion localization.
- This model was not deployed and does not replace the production system.

## 15. Reproducibility

- Script: `scripts/train_clean_dicm_melanoma_benign.py`
- Seed: 42
- Dataset: `C:\Projects\major-project\dataset\processed\dicm_clean_melanoma_benign`
- Output: `C:\Projects\major-project\models\experiments\clean_dicm_melanoma_benign`
- Production models, old dataset CSVs, and Experiments 1-3 were not modified.
- PH2 was not used.

