# Comparison: frozen 83.4% baseline vs class-weighted experiment

Baseline was not rerun. Quoted reference plus full-precision frozen summary.

| Metric | Baseline (quoted) | Baseline (frozen JSON) | Class-weighted | Delta vs frozen JSON |
|---|---:|---:|---:|---:|
| Accuracy | 0.834 | 0.8342 | 0.8272 | -0.0070 |
| Sensitivity | 0.815 | 0.8146 | 0.8511 | +0.0365 |
| Specificity | 0.849 | 0.8489 | 0.8095 | -0.0394 |
| Precision | 0.801 | 0.8008 | 0.7691 | -0.0317 |
| F1 | 0.808 | 0.8076 | 0.8080 | +0.0004 |
| ROC-AUC | 0.913 | 0.9127 | 0.9121 | -0.0005 |
| PR-AUC | 0.888 | 0.8885 | 0.8858 | -0.0027 |

Baseline ensemble 0.70/0.30 threshold 0.50. New ensemble 0.50/0.50 threshold 0.45.

Baseline CM TP=848 FP=211 TN=1185 FN=193.
New CM TP=886 FP=266 TN=1130 FN=155.

Verdict: trade-off, not a clear improvement

Some metrics improved and others worsened; class weighting shifted the operating point.

Not clinically validated. Scores are AI probability estimates.
