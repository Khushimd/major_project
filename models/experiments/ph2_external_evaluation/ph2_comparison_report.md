# OLD 92% MODEL vs NEW CLASS-WEIGHTED MODEL — PH2

Fixed 200-case PH2 cohort (40 melanoma, 160 nevi). Same cases for both models.
PH2 was used previously in this project; this is not a never-before-seen external set.

Old model: weights 0.75/0.25, threshold 0.30
Class-weighted: weights 0.50/0.50, threshold 0.45

| Metric | Old 92% model | Class-weighted model |
|---|---:|---:|
| Accuracy | 0.2050 | 0.2650 |
| Sensitivity | 1.0000 | 1.0000 |
| Specificity | 0.0063 | 0.0813 |
| Precision | 0.2010 | 0.2139 |
| F1 | 0.3347 | 0.3524 |
| ROC-AUC | 0.6573 | 0.8344 |
| PR-AUC | 0.4308 | 0.6304 |
| TP | 40 | 40 |
| TN | 1 | 13 |
| FP | 159 | 147 |
| FN | 0 | 0 |

Old model sensitivity 1.000 vs class-weighted 1.000; specificity 0.006 vs 0.081. False positives: old 159/160, class-weighted 147/160. Small PH2 melanoma n=40; treat metric differences as descriptive, not a ranking.

Not clinically validated. No model was modified from these results.
