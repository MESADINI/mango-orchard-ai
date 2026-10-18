Single-run ablations of the EfficientNet-B3 recipe on CV fold 0 (held-out n = 279 images; baseline = the CV fold-0 run). One run per variant: no seed variance is estimated.

| Variant | Accuracy | Macro-F1 [95% CI] | Δ macro-F1 | McNemar p | Best epoch |
|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 [1.000, 1.000] | — | — | 11 |
| basic_augmentation | 0.989 | 0.980 [0.954, 1.000] | -0.020 | 0.25 | 18 |
| img_size_256 | 0.996 | 0.992 [0.971, 1.000] | -0.008 | 1 | 18 |
| no_pretraining | 0.789 | 0.649 [0.584, 0.701] | -0.351 | 3.47e-18 | 27 |
