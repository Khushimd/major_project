# Frozen baseline validation FN analysis

Analysis-only experiment. No retraining, no threshold/ensemble retune,
no dashboard/production/baseline overwrite.

Validation FNs: 185. Recommended next intervention: B. Targeted augmentation.

Start here: `FN_vs_TP_analysis.md`, `fn_analysis_summary.json`.

Grad-CAM highlights image regions that contributed to the model's prediction. It provides a visual explanation of the regions influencing the model output and should not be interpreted as definitive evidence of malignancy.

NO test images were used for model-development decisions. Nothing was trained after this analysis.

