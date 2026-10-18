# Phase 2: Disease classifier study (reduced scope)

**Branch:** `enhance/real-data` · **Date:** 2026-10-17
**Exit criterion met:**
- 5-fold group CV and final models for three architectures;
- three single-fold ablations;
- calibration and OOD analysis;
- the selected model deployed with a SHA-verified copy.

Every number below is copied from `results/disease/` (`summary.json`, `tables/`, `figures/`, `latency.json`,
`deployed*.json`). Model card: `docs/models/disease_classifier.md`.

## 1. Scope (reduced on request)
- **Run:**
  - EfficientNet-B3, ResNet-50 and MobileNetV3-L: 5-fold group CV, giving 15 runs;
  - one final model per architecture;
  - three ablations of the EfficientNet-B3 recipe on CV fold 0.
- **Deferred, not run:**
  - ConvNeXt-T and Swin-T;
  - the original 15-run ablation grid;
  - the 288 px input size.

  These have no numbers and are listed in PROGRESS.md.

## 2. Protocol
- **CV fold k:**
  - held-out = fold k;
  - early stopping on fold (k+1) mod 5;
  - training on the other three folds plus the always-train groups.

  Folds are group-aware, so copies of the same leaf never cross folds. The test split is never touched by CV.
- **Final models:** train split → early stopping on val → one evaluation each on:
  - test (n = 239);
  - external (n = 381);
  - near-OOD (n = 1,493);
  - far-OOD (n = 300).
- **Logging:** each run writes `config.json`, `env.json` (git commit, package versions, GPU), `history.csv`,
  logits and `summary.json` with the checkpoint SHA-256.
- **Statistics:**
  - leaf-level bootstrap CIs;
  - exact McNemar on pooled out-of-fold predictions;
  - Wilcoxon on per-fold macro-F1 (n = 5, minimum attainable p = 0.0625);
  - Holm correction.

## 3. Results
| Model | CV accuracy | CV macro-F1 | Params | CPU latency |
|---|---|---|---|---|
| EfficientNet-B3 (selected) | 0.999 ± 0.002 | 0.998 ± 0.004 | 10.7 M | 48.7 ms |
| ResNet-50 | 0.999 ± 0.003 | 0.997 ± 0.007 | 23.5 M | 48.5 ms |
| MobileNetV3-L | 0.998 ± 0.002 | 0.996 ± 0.004 | 4.2 M | 19.7 ms |

No pairwise difference is significant (`tables/significance.md`; Holm-adjusted p ≥ 0.75). The selection rule was
the highest mean CV macro-F1, fixed before training. In-distribution, the three architectures are equivalent.

Final models (`tables/final_models.md`):

| Model | Test acc. | External macro-F1 [95% CI] | ECE test / external | AUROC near / far OOD | Flagged test / external |
|---|---|---|---|---|---|
| EfficientNet-B3 | 1.000 | 0.119 [0.083, 0.155] | 0.086 / 0.416 | 0.954 / 0.963 | 0.079 / 0.969 |
| ResNet-50 | 1.000 | 0.079 [0.056, 0.102] | 0.081 / 0.703 | 0.976 / 0.995 | 0.050 / 0.543 |
| MobileNetV3-L | 0.992 | 0.167 [0.118, 0.214] | 0.088 / 0.443 | 0.949 / 0.939 | 0.096 / 0.871 |

Ablations on fold 0 (n = 279; single runs, `tables/ablation.md`):

| Variant | Macro-F1 [95% CI] | Δ | McNemar p |
|---|---|---|---|
| baseline | 1.000 | — | — |
| basic augmentation | 0.980 [0.954, 1.000] | −0.020 | 0.25 |
| 256 px input | 0.992 [0.971, 1.000] | −0.008 | 1 |
| no pretraining | 0.649 [0.584, 0.701] | −0.351 | 3.5 × 10⁻¹⁸ |

**Only ImageNet pretraining has a measurable effect.**

## 4. Findings that change the claims
1. **External failure.** Near-perfect in-distribution accuracy does not transfer.
   - Every architecture scores ≤ 0.167 macro-F1 on the independent external set.
   - Bacterial Canker, which is 240 of the 381 external images, has recall 0.079 with EfficientNet-B3.
   - The paper must report this, not the test-split 1.000, as the field-relevant number.
2. **Low-confidence safety net.**
   - The threshold is set on val to keep 95% of images.
   - It flags 7.9% of test images but 96.9% of external images.
   - The app turns such a prediction into a "retake photo" message and excludes it from the OHI.
3. **Temperature not identifiable.**
   - Val accuracy is 100%, so the NLL-optimal temperature runs to the search bound (0.05).
   - Deploying it would make every prediction overconfident: external ECE 0.864 instead of 0.416.
   - A post-hoc rule, disclosed as such, deploys T = 1 when the fit is not identifiable.
4. **fp16 overflow.**
   - Under AMP autocast, EfficientNet-B3 produced non-finite logits on 5 far-OOD images.
   - All 21 runs were re-evaluated in fp32 from their checkpoints (SHA and image order verified;
     `logs/reeval_fp32.log`).
   - All evaluation is now fp32.

## 5. Deployment
- `ml/disease_detection/deploy.py` copies the selected `best.pth` with T, the threshold and the class list to
  `ml/disease_detection/weights/disease_classifier.pth`.
- It verifies SHA-256 `c6becbe0…8ff1`.
- `deployed_eval.json` re-evaluates the deployed file: test accuracy 1.000; external accuracy 0.110 and macro-F1
  0.119.

## 6. Code and tests
- **Code:**
  - `ml/disease_detection/`: `train.py`, `experiments.py`, `analysis.py`, `metrics.py`, `latency.py`,
    `deploy.py`, `evaluate.py`;
  - `ml/utils/experiment.py`: CUDA detection fix.
- **Tests:**
  - `ml/disease_detection/tests/`: metrics, CV partition, ablation partition, analysis tables, temperature
    identifiability, deployment.

The suite result is recorded in `results/tests/phase2_pytest.txt`.
