# Phase 4: Severity segmentation and explainability (partial)

**Branch:** `enhance/real-data` · **Date:** 2026-10-17

**Status:**
- **Done:** segmenter fixes; validation tooling; Grad-CAM deletion/insertion metrics.
- **Blocked:** severity validation and the pointing game. Both need ≥ 50 expert lesion masks, which must be drawn
  by a person.

  Instructions: `docs/annotation/SEVERITY_MASKS.md`. Fixed sample: `ml/datasets/annotation/severity_sample.csv`
  (60 test images, 12 per class, one per leaf).

## 1. Segmenter fixes (`ml/severity/severity.py`)
- **Sooty mould.** The black coating has low saturation and was missed by the brown-hue rule. It is now caught by
  marking tissue that is dark relative to the leaf: V < 0.55 × the median V of the leaf's green pixels, or
  V < 45. The rule is relative, so dark-green leaves are not marked.
- **Background and soil.**
  - The leaf region is the largest filled green component.
  - It grows through attached lesion-coloured tissue inside its convex hull, so lesions on the margin count.
  - If growth would add more than 10% of the image border, the leaf is treated as lying on a dark background and
    the growth falls back to the hull.
  - Paper, soil and shadows outside the leaf never count as lesion.
- **Resolution.** Images are downscaled to 512 px (longest side), so kernel sizes mean the same for every photo.
- **Known limitation, documented in the module:** the white powdery-mildew coating is not segmented as lesion.
- **Tests:** `ml/severity/tests/test_severity.py`, 8 synthetic cases: sooty patch, margin coating, brown spots,
  soil/shadow, dark background, no leaf, large image. The API keys are unchanged.

## 2. Validation tooling (`ml/severity/validate.py`)
- `sample`: draws the fixed 60-image sample.
- `convert`: converts LabelMe polygons to 0/128/255 masks.
- `evaluate`:
  - lesion IoU / Dice, leaf IoU, severity MAE (percentage points) and Spearman, all with bootstrap CIs;
  - writes `results/severity/{validation.json, summary.md, per_image.csv}`;
  - refuses to report with fewer than 50 masks.

**Results:** _pending: no masks yet (`results/severity/` does not exist)._

## 3. Grad-CAM faithfulness (`ml/explainability/faithfulness.py`)
- **Deletion / insertion AUC** (Petsiuk et al., 2018):
  - deployed classifier, MangoLeafBD test split, 20 steps;
  - removed pixels are filled with a blurred copy;
  - two baselines per image: a random ranking and a centred Gaussian (centre prior);
  - paired Wilcoxon tests.
- **Pointing game** (15 px tolerance): runs automatically once the masks exist.

**Results** (copied from `results/xai/faithfulness.md`; n = 239, mean ± SD):

| Curve | Grad-CAM | Random | Centre prior | Share of images where Grad-CAM beats random (p) | Share of images where Grad-CAM beats centre (p) |
|---|---|---|---|---|---|
| Deletion AUC (lower is better) | 0.606 ± 0.177 | 0.286 ± 0.114 | 0.629 ± 0.159 | 0.08 (2e-36) | 0.59 (0.00068) |
| Insertion AUC (higher is better) | 0.821 ± 0.100 | 0.285 ± 0.114 | 0.807 ± 0.103 | 1.00 (5.9e-41) | 0.62 (0.00014) |

**Interpretation:**
- **Versus the centre prior** (the fairer, contiguous baseline), Grad-CAM is more faithful on both curves. The
  effect is small and significant, but per image Grad-CAM wins only on about 60%.
- **Versus random ranking:**
  - Grad-CAM is far better on insertion.
  - On deletion it looks "worse". Scattered random deletion turns the image into out-of-distribution noise, so the
    prediction collapses whatever the pixels were (Hooker et al., 2019). Random deletion is therefore not a
    meaningful lower bound.
- **What the paper may claim:** Grad-CAM highlights regions modestly more decision-relevant than a centre prior.
- **What it may not claim:** that the heatmaps localise lesions. That needs the pointing game, which is blocked on
  the masks.
