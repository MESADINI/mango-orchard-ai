# Phase 3: Pest detector

**Branch:** `enhance/real-data` · **Date:** 2026-10-17

All numbers below are copied from `results/pest/` (`models.md`, `*/summary.json`, `deployed.json`). Model card:
`docs/models/pest_detector.md`.

## 1. Protocol (`ml/pest_detection/train.py`)
- **Data:** `ml/datasets/processed/pest/`, built in Phase 1.
  - 2,559 images in 1,400 groups;
  - group-aware split with no overlap: train 1,786, val 394, test 379.
- **Classes:** Fruit Fly, Mealy Bug, Scale Insect. Each class comes from **one non-mango source**:
  - Fruit Fly: grayscale trap photos;
  - Mealy Bug: grape and stock photos;
  - Scale Insect: coffee leaves.

  Class and image domain are therefore confounded.
- **Training:**
  - COCO-pretrained start; the official Ultralytics weights are pinned by SHA-256 in `PRETRAINED`;
  - 640 px, SGD (lr 0.01), up to 50 epochs, early stopping on val (patience 15);
  - seed 42, `deterministic=True`;
  - Ultralytics 8.3.40, upgraded from 8.2.18 for YOLO11 and recorded in `requirements-ml.lock` and
    `backend/requirements.txt`.
- **Evaluation:** once on val and once on test (conf 0.001, IoU 0.7). Every run writes `config.json`, `env.json`
  and `summary.json`, the last with the checkpoint SHA-256.
- **Selection and deployment:**
  - the model with the highest **val** mAP50-95 is selected; the test split is reported only;
  - `deploy` copies that run's `best.pt` by explicit path, which replaces the old "first `best.pt` found by a
    glob" behaviour;
  - it verifies the SHA-256 and writes `pest_detector.json` with the class list and `host_verified` flags.

## 2. Results
| Model | Params | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test AP50-95: Fruit Fly / Mealy Bug / Scale Insect |
|---|---|---|---|---|---|
| YOLOv8n | 3.0 M | 0.533 | 0.743 | 0.519 | 0.539 / 0.118 / 0.899 |
| **YOLO11n** (deployed) | 2.6 M | 0.537 | 0.744 | 0.520 | 0.542 / 0.121 / 0.896 |

- **The two architectures are equivalent.** Each was trained once, so the 0.001 test difference carries no
  meaning.
- **Mealy Bug fails:** test AP50 0.26, recall about 0.33. Its source mixes grape bunches with watermarked stock
  photos of mealybugs on assorted plants.
- **Scale Insect and Fruit Fly score high,** but each class is a single homogeneous source: coffee leaves and
  grayscale traps. The detector can partly separate the classes by background and colour. These numbers do not
  estimate performance on mango trees.
- **Deployed:** YOLO11n, `ml/pest_detection/weights/pest_detector.pt`, SHA-256 `dfd2d0e2…4315`
  (registry version `dfd2d0e2db47`).

## 3. Not completed
- **YOLOv8s:** stopped at epoch 12 of 50, when the system ran low on memory and the session ended the job. It was
  not restarted. The partial output was moved out of `results/`. No numbers exist.
- **RT-DETR:** not run.

Both are listed in PROGRESS.md under *Deferred, not run*.

## 4. Consequences in the app
- Every class is `host_verified: false` (`ml/datasets/pest_source_review.json`). The detector shows its boxes with
  an "UNVERIFIED: model not trained on mango" badge.
- The pest component is excluded from the Orchard Health Index (OHI), and no pest treatment is recommended from
  these detections. The rules are in `docs/OHI_METHODOLOGY.md` §3B; the code landed in the Phase 5 commit.
- Mango Hopper and Stem Borer remain gated at < 100 samples and are not trained.
- Converting the detector into evidence for mango orchards needs annotated pest images from mango trees.

## 5. Tests
- `ml/pest_detection/tests/test_train.py`:
  - host review → flags, and the repository's sources are not mango;
  - selection ignores the test split;
  - SHA-verified deployment rejects a tampered checkpoint.
- `test_pest_detection.py`: `host_verified` defaults and metadata loading.
- `ml/tests/test_e2e_real_models.py`: the real detector is flagged host-unverified and the pest component is
  missing.
