# Phase 1: Real, leakage-controlled datasets

**Branch:** `enhance/real-data` · **Date:** 2026-10-17
**Exit criterion met:** `python tasks.py data` rebuilds every dataset from scratch (verified SHA-256 downloads)
and writes `results/data_stats.json`. All numbers below are copied from `results/data/*.json`.

## 1. Disease dataset (MangoLeafBD), card: `docs/datasets/mangoleafbd.md`

**Source.** Kaggle credentials were unavailable, so the data comes from the Hugging Face mirror
`Project-AgML/MangoLeafBD_disease_classification`, pinned to a revision with a verified SHA-256. All 4,000 images
match the SHA-256 list on the official Mendeley record (4000/4000), and each Mendeley folder maps to exactly one
class (`results/data/mangoleafbd_download.json`). License: CC BY-NC 3.0. 8 classes × 500 images; 21 exact duplicate
files leave 3,979 unique images.

**Leaf Spot → 5 classes.** MangoLeafBD has no Leaf Spot class, and no openly licensed, mango-specific, labelled
Leaf Spot set was found. The model is 5-class everywhere: `DISEASE_CLASSES` in `inference.py`, OHI virulence table,
train script, README/docs, and the paper drafts (marked reference-only). A test (`test_leaf_spot_is_not_a_model_class`)
guards this, and the old `leaf_spot_sample.jpg` was removed.

**Near-duplicate grouping.** The authors state that only about 1,800 of the 4,000 images are distinct leaves; the rest
are zoomed or rotated copies. A random split would leak copies of the same leaf into the test set.

| Attempt | Finding (evidence) | Decision |
|---|---|---|
| pHash, dihedral-invariant | 250 identical pairs at distance 0 (`n_phash_edges`), none cross-class; during exploration, different leaves collided from distance 2–4 (visual check, not an evidence file) | keep, distance 0 only |
| DINOv2 cosine threshold | Many same-class pairs at cosine ≥ 0.96 are *different* leaves (`results/data/dedup_review/cos_*_same.jpg`); thresholding merges a class into giant groups | **rejected** as a link criterion; used only for retrieval |
| DINOv2 top-k retrieval + SIFT/RANSAC similarity transform | Copies made by zoom, rotation or mirroring keep the vein and lesion geometry | **adopted** |
| …first version | Many-to-one matches let a 1000-keypoint image "match" a ~20-keypoint image with hundreds of inliers (found in an exploratory probe of the external set; superseded run) | fixed: one-to-one matches; regression test added |

Calibration (`python ml/datasets/build_disease_dataset.py --calibrate`, `results/data/disease_dedup_calibration.json`):
- With one-to-one matching, **no cross-class pair is ever linked**: 0 of 4,196 cross-class candidates at k=40, and
  0 of 14,474 at k=80, for every threshold from 10 inliers up. Cross-class pairs cannot be the same leaf, so this
  bounds the false-link rate.
- **Group counts by k, at 12 inliers:** k=10 → 1,821; k=40 → 1,783; k=80 → 1,773 (< 1% change, converged).
- **Chosen parameters:** k = 40, ≥ 12 inliers. At this threshold the 12–18 inlier band is visually almost all the
  same leaf (`inliers_12-18_same.jpg`), and **1,783 groups** match the authors' "around 1,800 distinct leaves".
  Erring towards linking is deliberate: a missed link leaks, a false link only costs split flexibility.

**Split.** A single Sooty Mould leaf has 190 copies. Groups larger than 5% of their class (17 groups; 273 Bacterial
Canker, 265 Powdery Mildew and 328 Sooty Mold images) go to train and are never held out in CV (`cv_fold = -2`).
Otherwise one leaf would dominate a class's validation or test metrics (observed in the first build: 237 Sooty
Mold images in val vs 194 in train). The remaining groups are split with StratifiedGroupKFold (seed 42).

| | train | val | test |
|---|---|---|---|
| Images | 2,007 | 240 | 239 |
| Groups | 767 | 159 | 159 |

Distinct leaves (groups) per class: Anthracnose 248, Bacterial Canker **76**, Powdery Mildew 167, Sooty Mold 107,
Healthy 487 (`groups_per_class`). Bacterial Canker's 500 images come from only 76 leaves, so its effective sample
size is much smaller than its image count.

Group overlap between splits is 0/0/0. The 5 CV folds over train+val (excluding the always-train groups) hold
279, 248, 264, 280 and 310 images. The per-class table is in the card.
**Implication for Phase 2:** the held-out test set is small (239 images, 159 leaves), so 5-fold group CV is the
primary estimate, reported as mean ± SD, with the test set as a single confirmatory evaluation.

## 2. External test set, card: `docs/datasets/external.md`

Source: Rahman, Hasan & Mojumdar (2024), Mendeley 7ghdbftp54, CC BY 4.0. The 1.4 GB archive was SHA-256 verified;
only `process data.zip` is used, and `Augmented Image.zip` (13,504 images) is skipped.

**Provenance finding.** 972 of the 1,353 candidates have exactly MangoLeafBD's pixel sizes and, on inspection, are
MangoLeafBD leaves with a blacked-out background. The leakage guard independently confirms 581 of them as geometric
matches to specific MangoLeafBD images, and **all Healthy images** are in this group. Images at MangoLeafBD sizes are
excluded (format rule) and listed in `excluded.csv`. Of the 381 native-resolution images, **0** match MangoLeafBD.

**Result:** 381 images: Anthracnose 26, Bacterial Canker 240, Powdery Mildew 52, Sooty Mold 63, **no Healthy**. The set
therefore measures per-class recall under a camera/orchard shift; it cannot measure false alarms on healthy leaves.

## 3. Pest dataset, card: `docs/datasets/pest.md`

Downloaded through the Roboflow REST export API: the SDK would have changed numpy, opencv and certifi, and the key is
read from the environment and never printed. Extraction uses short file names because of the Windows path-length
limit. The scale-insect export uses polygon labels, which are converted to boxes.

**Host-plant review** (`results/data/pest_review/*.jpg`, recorded in `ml/datasets/pest_source_review.json`):

| Source | Shows | Host |
|---|---|---|
| `mango-fruit-fly` v4 | grayscale flies in trap dishes; its `Pest` class is ants, snails and bees (dropped) | trap, no plant |
| `mealybug-b8wfq` v8 | grape bunches, vine bark, watermarked stock photos | grape |
| `hemispherical-scale-insect` v1 | *Saissetia coffeae* on coffee leaves next to `Coffee Leaf Rust` (dropped) | coffee |

**No public source shows pests on mango.** Excluding all of them would leave no pest module, which would change the
project idea. Following the brief ("flag e.g. the grape mealybug"), the set is built with `--include-non-mango` as
**flagged proxy data**, and the flag is recorded in the evidence. The default, strict build (without the flag)
yields zero images.

Processing details:
- Only the named source classes are kept.
- Images whose every box belonged to a dropped class are removed, since they may hide unlabelled pests (2,973 /
  28 / 10).
- Exact duplicates are removed (917 within the fruit-fly export).
- Roboflow augmentation stems and pHash-identical images are grouped, and the split is group-aware 70/15/15.

**Result:** 2,559 images in 1,400 groups and 12,048 boxes: Fruit Fly 907 images / 7,937 boxes, Mealy Bug 853 / 2,532,
Scale Insect 799 / 1,579. 286 images are true background, and group overlap between splits is 0.
Rule 3 is unchanged: Mango Hopper and Stem Borer can never enter the set.

**Consequences, carried into Phases 3 and 6:**
- Every pest class carries `host_verified: false`.
- The UI shows "trained on non-mango images; confirm by inspection".
- The paper reports pest results as proxy-domain results only.

## 4. Field images

`ml/datasets/field/README.md` and `labels_template.csv` describe how to add your own orchard photos (5 classes plus
`Other` for rejection tests, labeller, confidence, device, location) and pest photos. The images are git-ignored.

## 5. Code, tests, environment

- **Removed:** `phase1_disease.py`, `phase1_pest.py`, `download_datasets.py`, `prepare_pest_data.py` and
  `generate_sample_data.py`. These wrote synthetic cartoon images into `processed/`, where they could be mistaken
  for data.
- **New scripts:** `download_mangoleafbd.py`, `build_disease_dataset.py`, `build_external_testset.py`,
  `build_pest_dataset.py`, `dedup.py`, `fetch.py` (resumable 64-way range download; Mendeley serves about 3 KB/s per
  connection), `data_stats.py` and `make_cards.py`. `make_cards.py` fills the card tables from the evidence files,
  so the cards cannot drift from the data.
- **Tests:** 45 → **90 passed, 0 skipped** (`results/tests/phase1_pytest.txt`). The new tests cover unit tests for
  every helper, the geometric verification on synthetic zoomed, rotated and mirrored copies, and the one-to-one
  regression. Data-integrity tests check:
  - no SHA or group overlap between splits;
  - CV folds respect groups and exclude test;
  - the 70/15/15 proportions;
  - the external set excludes MangoLeafBD-format images and matches;
  - pest labels are valid and never gated.
- **Environment:** torch is now 2.3.0+cu121 (the RTX 3050 is used; DINOv2 embeddings take about 4 min). Also pinned:
  pyarrow 17.0.0 (18+ needs numpy 2) and huggingface_hub 0.23.4 (2.x breaks timm 1.0.3). Recorded in
  `requirements-ml.txt` and `requirements-ml.lock`; `pip check` is clean.

## 6. Known limitations / open items

- Grouping is an estimate. Copies of feature-poor healthy leaves may escape linking (487 groups from 500 Healthy
  images, against 76 from 500 Bacterial Canker images), so a small residual leakage cannot be excluded.
  Phase 2 should report Healthy separately.
- The external set has no Healthy class, and Bacterial Canker dominates it.
- The pest data is proxy data (not mango). Mango field images are needed before any pest claim.
- The host-plant review was done by the AI assistant from contact sheets and should be confirmed by a person.
