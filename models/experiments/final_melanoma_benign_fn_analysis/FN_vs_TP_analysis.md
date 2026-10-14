# Validation false-negative vs true-positive analysis

Research analysis only. **Not clinically validated.** Scores are AI probability estimates.

## 1. Objective

Understand why the frozen 83.42% baseline misses melanoma on the **validation** set
before deciding whether another targeted training experiment is justified.

## 2. Frozen model configuration

EfficientNetB0 0.70 + MobileNetV2 0.30, threshold 0.50, frozen `final_melanoma_benign` splits.
Scores reused from the frozen baseline validation inference artifacts after verifying the CM.

## 3. Validation cohort

2438 images; 1043 melanoma; 1395 benign.

## 4. Number of validation FNs

185 (label=1 and ensemble_prob < 0.50). Reproduced CM TP=858 TN=1157 FP=238 FN=185.

## 5. FN score distribution

{
  "tp": {
    "n": 858,
    "mean": 0.8737913138018205,
    "median": 0.9485057920217514,
    "std": 0.14848670423598748,
    "min": 0.502732554078102,
    "max": 0.9999374330043792,
    "q1": 0.782503954321146,
    "q3": 0.9925177022814751
  },
  "fn": {
    "n": 185,
    "mean": 0.3450553738627884,
    "median": 0.3653564184904098,
    "std": 0.10255661556458494,
    "min": 0.1098253719508647,
    "max": 0.4993993312120437,
    "q1": 0.2697331964969635,
    "q3": 0.4338692307472228
  },
  "fn_bins": [
    {
      "range": "0.00\u20130.10",
      "low": 0.0,
      "high": 0.1,
      "count": 0,
      "percent": 0.0
    },
    {
      "range": "0.10\u20130.20",
      "low": 0.1,
      "high": 0.2,
      "count": 21,
      "percent": 11.351351351351353
    },
    {
      "range": "0.20\u20130.30",
      "low": 0.2,
      "high": 0.3,
      "count": 40,
      "percent": 21.62162162162162
    },
    {
      "range": "0.30\u20130.40",
      "low": 0.3,
      "high": 0.4,
      "count": 58,
      "percent": 31.351351351351354
    },
    {
      "range": "0.40\u20130.50",
      "low": 0.4,
      "high": 0.5,
      "count": 66,
      "percent": 35.67567567567568
    }
  ],
  "fn_borderline_ge_0.40": 66,
  "fn_very_low_lt_0.20": 21
}

## 6. EfficientNet vs MobileNet disagreement

{
  "fn_both_below_0.50": 133,
  "fn_efficientnet_ge_0.50_mobilenet_lt_0.50": 48,
  "fn_mobilenet_ge_0.50_efficientnet_lt_0.50": 4,
  "fn_mean_abs_score_diff": 0.19800131929729728,
  "tp_mean_efficientnet": 0.9093759533799534,
  "tp_mean_mobilenet": 0.7907604858275058,
  "fn_mean_efficientnet": 0.397830015027027,
  "fn_mean_mobilenet": 0.22191454545945943,
  "tp_mean_ensemble": 0.8737913138018205,
  "fn_mean_ensemble": 0.3450553738627884
}

## 7. Metadata findings

{
  "all_validation_melanoma_source": "ISIC-DICM-17K",
  "source_contrast_among_melanoma": "None. Every validation melanoma row is ISIC-DICM-17K, so FN vs TP cannot show a source/domain split.",
  "patient_id_missing_fn_percent": 45.40540540540541,
  "lesion_id_missing_fn_percent": 31.891891891891895,
  "age_fn": {
    "n": 179,
    "mean": 58.324022346368714,
    "median": 60.0,
    "std": 15.533075746831381,
    "min": 15.0,
    "max": 85.0,
    "q1": 47.5,
    "q3": 70.0
  },
  "age_tp": {
    "n": 830,
    "mean": 61.60843373493976,
    "median": 60.0,
    "std": 14.8262043050106,
    "min": 25.0,
    "max": 85.0,
    "q1": 50.0,
    "q3": 75.0
  },
  "sex_fn_percent": {
    "male": 49.19,
    "female": 47.57,
    "NA": 3.24
  },
  "sex_tp_percent": {
    "male": 60.72,
    "female": 35.78,
    "NA": 3.5
  },
  "site_frequency_notes": [
    "TPs were more frequent in anterior torso (FN share lower by 12.1 pp)",
    "TPs were more frequent in head/neck (FN share lower by 5.7 pp)",
    "FNs were more frequent in lower extremity (23.2% of FN vs 16.6% of TP)",
    "TPs were more frequent in palms/soles (FN share lower by 6.5 pp)",
    "FNs were more frequent in posterior torso (19.5% of FN vs 9.3% of TP)",
    "FNs were more frequent in upper extremity (19.5% of FN vs 12.2% of TP)"
  ],
  "missingness": {
    "patient_id": {
      "fn_percent": 45.40540540540541,
      "tp_percent": 83.1002331002331
    },
    "lesion_id": {
      "fn_percent": 31.891891891891895,
      "tp_percent": 15.268065268065268
    },
    "sex": {
      "fn_percent": 3.2432432432432434,
      "tp_percent": 3.4965034965034967
    },
    "age_approx": {
      "fn_percent": 3.2432432432432434,
      "tp_percent": 3.263403263403263
    },
    "anatom_site": {
      "fn_percent": 6.486486486486487,
      "tp_percent": 7.226107226107226
    }
  }
}

Described as frequency differences only. Not causal.

## 8. Image-quality findings

{
  "pixels_analyzed": "Frozen baseline 224x224 cache (what the model received). Original file size recorded separately when the path existed.",
  "lesion_area": "Lesion area could not be reliably quantified from the available data.",
  "fn": {
    "brightness": {
      "n": 185,
      "mean": 155.2991187018317,
      "median": 155.0108184814453,
      "std": 24.280741534384777,
      "min": 73.1744384765625,
      "max": 219.69802856445312,
      "q1": 139.3317108154297,
      "q3": 171.9432830810547
    },
    "contrast": {
      "n": 185,
      "mean": 24.70281228761415,
      "median": 25.097280502319336,
      "std": 10.614771293744418,
      "min": 6.612983703613281,
      "max": 66.97686004638672,
      "q1": 15.872515678405762,
      "q3": 31.086624145507812
    },
    "color_variation": {
      "n": 185,
      "mean": 25.458038201203216,
      "median": 25.72850227355957,
      "std": 10.463949371787816,
      "min": 7.025712490081787,
      "max": 67.67637634277344,
      "q1": 16.495208740234375,
      "q3": 31.887060165405273
    },
    "orig_width": {
      "n": 185,
      "mean": 2656.0162162162164,
      "median": 2801.0,
      "std": 1552.9998826803464,
      "min": 600.0,
      "max": 6668.0,
      "q1": 1024.0,
      "q3": 3264.0
    },
    "orig_height": {
      "n": 185,
      "mean": 1894.5837837837837,
      "median": 2000.0,
      "std": 1046.4772088529442,
      "min": 450.0,
      "max": 4459.0,
      "q1": 1024.0,
      "q3": 2448.0
    }
  },
  "tp": {
    "brightness": {
      "n": 858,
      "mean": 137.11213429324278,
      "median": 141.3959503173828,
      "std": 31.141367022350032,
      "min": 44.27350616455078,
      "max": 222.20616149902344,
      "q1": 120.19705200195312,
      "q3": 158.60992813110352
    },
    "contrast": {
      "n": 858,
      "mean": 48.01523640216925,
      "median": 44.71780204772949,
      "std": 21.587251994579706,
      "min": 10.125974655151367,
      "max": 97.32463073730469,
      "q1": 30.65618658065796,
      "q3": 65.52938270568848
    },
    "color_variation": {
      "n": 858,
      "mean": 48.66259499196406,
      "median": 45.42901611328125,
      "std": 21.481312523603535,
      "min": 11.219943046569824,
      "max": 97.54864501953125,
      "q1": 31.430131912231445,
      "q3": 66.33134651184082
    },
    "orig_width": {
      "n": 858,
      "mean": 1386.6025641025642,
      "median": 1024.0,
      "std": 1171.0349766507868,
      "min": 371.0,
      "max": 6708.0,
      "q1": 1024.0,
      "q3": 1024.0
    },
    "orig_height": {
      "n": 858,
      "mean": 1192.6958041958042,
      "median": 1024.0,
      "std": 802.7805947577441,
      "min": 441.0,
      "max": 4499.0,
      "q1": 1024.0,
      "q3": 1024.0
    }
  }
}

Lesion area could not be reliably quantified from the available data.
No clinical segmentation masks were used or invented.

## 9. Grad-CAM findings

Grad-CAM highlights image regions that contributed to the model's prediction. It provides a visual explanation of the regions influencing the model output and should not be interpreted as definitive evidence of malignancy.

{
  "disclaimer": "Grad-CAM highlights image regions that contributed to the model's prediction. It provides a visual explanation of the regions influencing the model output and should not be interpreted as definitive evidence of malignancy.",
  "model": "efficientnetb0_final_baseline.keras",
  "n_fn": 20,
  "n_tp": 20,
  "mean_center_share_fn": 0.49604793041944506,
  "mean_center_share_tp": 0.4005941728129983
}

## 10. Main observed failure patterns

{
  "borderline_score_fn": 66,
  "very_low_score_fn": 21,
  "model_agreement_fn": 133,
  "model_disagreement_fn": 52,
  "possible_image_quality_related_fn": 89,
  "possible_source_domain_related_fn": 0,
  "possible_small_subtle_lesion_fn": 0,
  "no_clear_identifiable_pattern": 46,
  "notes": {
    "source_domain": "Not identifiable among validation melanoma (single source).",
    "small_subtle_lesion": "Not used. Lesion area could not be reliably quantified from the available data."
  }
}

## 11. Limitations

- Validation only; test FNs were not inspected for development decisions.
- All validation melanoma is ISIC-DICM-17K, so melanoma FN vs TP cannot show a source contrast.
- Patient IDs are often missing.
- Image stats are from the 224×224 cached pixels the model saw.
- Grad-CAM is not segmentation and is not medical proof.

## 12. Recommended ONE next intervention

**B. Targeted augmentation**

Most validation FNs are missed by both models. Threshold search and moderate class weighting already failed to keep an acceptable spec/prec band. A next training test, if any, should target hard melanoma appearance (subtle/low-contrast/borderline scores) via training-time augmentation only — not TTA-max and not another weight grid. Do not train it in this run.

This analysis did not train that intervention.

## 13. Test-set statement

NO test images were used for model-development decisions.
The frozen test FN count (193) is quoted only as previously published baseline context.

