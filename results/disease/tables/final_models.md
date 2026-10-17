Final models (train split, early stopping on val), evaluated once. CIs: leaf-level bootstrap. T: temperature fitted on val; '(n.i.)' = not identifiable (val perfectly separated, or the fit on the search bound), so T = 1 is deployed (post-hoc rule, disclosed). Low-confidence flag: top probability below the value that keeps 95% of val images; flagged images get a 'retake photo' message.

| Model | Test acc. [95% CI] | Test macro-F1 [95% CI] | T fitted → deployed | ECE test / external | External macro-F1 (4 cls) [95% CI] | AUROC near / far OOD | FPR@95 near / far | Flagged low-confidence: test / external |
|---|---|---|---|---|---|---|---|---|
| efficientnet_b3 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.05 (n.i.) → 1.00 | 0.086 / 0.416 | 0.119 [0.083, 0.155] | 0.954 / 0.963 | 0.24 / 0.08 | 0.079 / 0.969 |
| resnet50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.05 (n.i.) → 1.00 | 0.081 / 0.703 | 0.079 [0.056, 0.102] | 0.976 / 0.995 | 0.09 / 0.03 | 0.050 / 0.543 |
| mobilenetv3_large | 0.992 [0.979, 1.000] | 0.988 [0.964, 1.000] | 0.05 (n.i.) → 1.00 | 0.088 / 0.443 | 0.167 [0.118, 0.214] | 0.949 / 0.939 | 0.30 / 0.26 | 0.096 / 0.871 |
