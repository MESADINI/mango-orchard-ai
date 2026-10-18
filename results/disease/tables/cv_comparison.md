5-fold group cross-validation (mean ± SD over folds). Source: results/disease/summary.json

| Model | Accuracy | Macro-F1 | Balanced acc. | Macro AUC (OvR) | Params (M) | CPU latency (ms) |
|---|---|---|---|---|---|---|
| efficientnet_b3 **(selected)** | 0.999 ± 0.002 | 0.998 ± 0.004 | 0.999 ± 0.002 | 1.000 ± 0.000 | 10.7 | 48.7 |
| resnet50 | 0.999 ± 0.003 | 0.997 ± 0.007 | 0.998 ± 0.005 | 1.000 ± 0.000 | 23.5 | 48.5 |
| mobilenetv3_large | 0.998 ± 0.002 | 0.996 ± 0.004 | 0.996 ± 0.005 | 1.000 ± 0.000 | 4.2 | 19.7 |
