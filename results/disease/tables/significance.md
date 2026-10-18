Paired tests on CV results. McNemar: pooled out-of-fold predictions (b = first right/second wrong, c = reverse). Wilcoxon: per-fold macro-F1, n = 5 (minimum attainable p = 0.0625).

| Pair | McNemar b/c | McNemar p (Holm) | Wilcoxon p (Holm) |
|---|---|---|---|
| efficientnet_b3 vs resnet50 | 2/1 | 1.0000 (1.0000) | 1.0000 (1.0000) |
| efficientnet_b3 vs mobilenetv3_large | 3/1 | 0.6250 (1.0000) | 0.2500 (0.7500) |
| resnet50 vs mobilenetv3_large | 3/2 | 1.0000 (1.0000) | 1.0000 (1.0000) |
