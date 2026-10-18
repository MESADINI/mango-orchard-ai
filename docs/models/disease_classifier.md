# Model card: mango leaf disease classifier

All numbers are copied from `results/disease/` (`summary.json`, `tables/*.md`, `deployed_eval.json`). Registry
entry: `ml/model_registry.json`.

## Model
| | |
|---|---|
| Architecture | EfficientNet-B3 (`timm` `efficientnet_b3.ra2_in1k`, ImageNet-pretrained), 10.7 M parameters |
| Task | 5-class leaf image classification: Anthracnose, Bacterial Canker, Powdery Mildew, Sooty Mold, Healthy |
| Deployed weights | `ml/disease_detection/weights/disease_classifier.pth`, SHA-256 `c6becbe0…8ff1` (version `c6becbe02b05`) |
| Source run | `results/disease/final/efficientnet_b3/best.pth` (SHA-256 `692d5315…967f`), best epoch 10 of 16 |
| Selection | highest mean 5-fold CV macro-F1 among the study architectures (EfficientNet-B3, ResNet-50, MobileNetV3-L) |
| Training | 224 px, strong augmentation, AdamW (lr 2e-4, weight decay 0.05), 2 warm-up epochs, label smoothing 0.1, class weights, batch 32, ≤ 30 epochs with early stopping (patience 6) on val, seed 42, AMP for training only |
| Evaluation | fp32 (AMP inference overflowed to non-finite logits on some out-of-distribution images) |
| Calibration | temperature T = 1.0 (see *Calibration*); low-confidence threshold 0.8152 = top probability that keeps 95% of val images |
| CPU latency | 48.7 ms per image (batch 1, idle machine; `results/disease/latency.json`) |

## Data
- **Training data:** MangoLeafBD (CC BY-NC 3.0; card `docs/datasets/mangoleafbd.md`).
  - Five of its eight classes are used.
  - Near-duplicate copies of the same leaf are grouped, and every split is group-aware: train 2,007, val 240,
    test 239 images.
- **External test set:** 381 independent images of four classes (no Healthy; `docs/datasets/external.md`).
- **Out-of-distribution (OOD) sets:**
  - near: 1,493 MangoLeafBD images of the three classes the model does not know;
  - far: 300 pest-dataset images.

## Results
| Evaluation | Accuracy | Macro-F1 [95% CI] | Notes |
|---|---|---|---|
| 5-fold group CV (mean ± SD) | 0.999 ± 0.002 | 0.998 ± 0.004 | in-distribution |
| Test split (n = 239) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | in-distribution; 7.9% flagged low-confidence |
| **External set (n = 381)** | **0.110** | **0.119 [0.083, 0.155]** | **fails**; 96.9% flagged low-confidence |

- In-distribution, ResNet-50 and MobileNetV3-L are statistically indistinguishable from EfficientNet-B3: every
  McNemar and Wilcoxon p ≥ 0.25 before Holm correction (`tables/significance.md`).
- Per-class recall on the external set (`tables/external_selected.md`):

| Class | Recall |
|---|---|
| Anthracnose | 0.615 |
| Bacterial Canker | 0.079 |
| Powdery Mildew | 0.000 |
| Sooty Mold | 0.111 |

- OOD detection with max-softmax probability:
  - AUROC: near 0.954, far 0.963;
  - FPR at 95% TPR: near 0.24, far 0.08.

## Calibration
- Every val image is classified correctly, so the temperature that minimises val NLL runs to the search bound
  (fitted T = 0.05) and is not identifiable.
- A rule added after the first results (post-hoc, and disclosed as such) deploys T = 1 whenever the fit is not
  identifiable.
- Test ECE at T = 1 is 0.086; on the external set it is 0.416.

## Intended use and limits
- **Use:** decision support for a mango leaf photo taken like MangoLeafBD's (single leaf, plain background). Any
  image below the low-confidence threshold gets a "retake photo" message, and its disease result is left out of
  the Orchard Health Index (OHI).
- **Do not use as evidence of field accuracy.** Near-perfect in-distribution scores did not transfer: macro-F1 on
  the independent external set is 0.119. The low-confidence flag catches most of these images (96.9%), but it is
  a safety net, not a fix. Collecting field images from Indian orchards is the necessary next step.
- **No Leaf Spot class:** there is no licensed training data for it.
- **Not run:**
  - ConvNeXt-T and Swin-T;
  - multi-seed ablations;
  - the 288 px input size.

  See PROGRESS.md, *Deferred, not run*.
