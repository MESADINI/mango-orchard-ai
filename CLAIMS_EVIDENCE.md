# Claims and evidence

Every claim the README, docs or paper may make, with the file that supports it. A statement missing from the
"Supported" table must not be made. Numbers are copied from the files named; regenerate tables with
`python tasks.py paper-tables`.

## Supported
| # | Claim | Evidence |
|---|---|---|
| 1 | Disease splits are leakage-controlled: copies of the same leaf are grouped, and no group crosses splits | `results/data_stats.json`, `results/data/`, `docs/phases/PHASE_1.md`, `ml/datasets/tests/test_data_integrity.py` |
| 2 | 5-fold group CV macro-F1 of the selected EfficientNet-B3 is 0.998 ± 0.004; ResNet-50 and MobileNetV3-L are not significantly different | `results/disease/summary.json`, `tables/cv_comparison.md`, `tables/significance.md` |
| 3 | Test-split accuracy is 1.000 (n = 239, in-distribution) | `results/disease/tables/final_models.md`, `deployed_eval.json` |
| 4 | **External-set macro-F1 is 0.119 [0.083, 0.155]** (n = 381): the model does not transfer | same files; `tables/external_selected.md` |
| 5 | The low-confidence rule flags 7.9% of test images and 96.9% of external images ("retake photo") | `results/disease/summary.json` |
| 6 | Removing ImageNet pretraining lowers fold-0 macro-F1 from 1.000 to 0.649 (single runs) | `results/disease/tables/ablation.md` |
| 7 | The deployed temperature is 1.0, because the validation fit was not identifiable (post-hoc rule, disclosed) | `results/disease/summary.json` (`temperature_*`) |
| 8 | The deployed weights match the registry (SHA-256) | `ml/model_registry.json`, `python ml/model_registry.py verify`, `GET /api/models` |
| 9 | Pest detector YOLO11n: test mAP50-95 0.520 on proxy data. Mealy Bug AP50 is 0.26 | `results/pest/models.md`, `results/pest/deployed.json` |
| 10 | No pest class was trained on mango images (all `host_verified: false`); pest results are excluded from the OHI | `ml/datasets/pest_source_review.json`, `ml/pest_detection/weights/pest_detector.json`, `ml/tests/test_e2e_real_models.py` |
| 11 | Grad-CAM is modestly more faithful than a centre prior: deletion AUC 0.606 vs 0.629, insertion 0.821 vs 0.807, better on about 60% of images | `results/xai/faithfulness.md` |
| 12 | The OHI never scores an unassessed component as healthy; with no image component there is no OHI | `ml/severity/ohi.py`, `ml/severity/tests/test_ohi.py`, `ml/tests/test_full_pipeline.py` |
| 13 | OHI ranking is robust (Spearman ρ ≥ 0.993 at ±20% on one weight), but tiers change for up to 12.1% of scenarios (42.3% median with uniform weights) | `results/ohi/sensitivity.md` |
| 14 | In 2015–2024 NASA POWER data for Krishnagiri, the NHM bulletin's "High" conditions occur on 0.0 days/year; the pre-audit rules flagged 216.2 days/year | `results/forecaster/backtest.json`, `monthly.md` |
| 15 | Each treatment is linked and quoted from TNAU / NHM and checked against the CIB&RC banned list (31.07.2026) | `docs/sources.md`, `ml/forecasting/recommendations_db.json`, `ml/forecasting/tests/test_recommendations.py` |
| 16 | 21 of 26 references verify against Crossref / arXiv; the other 5 are web, software or JSTOR sources checked by hand | `results/paper/references_check.json` |
| 17 | Median POST /api/analyze latency on CPU is 404 ms for a dual scan (disease 345 ms, pest 99 ms), network weather excluded | `results/api/latency.md`, `latency.json` |

## Not supported (do not claim)
| Claim | Why not |
|---|---|
| Field or orchard accuracy of the disease model | The only independent evaluation (claim 4) shows failure |
| The pest detector works on mango trees | No mango images in training or test |
| Lesion-area accuracy | No expert masks yet (`docs/annotation/SEVERITY_MASKS.md`) |
| Grad-CAM localises lesions | The pointing game needs the masks |
| The OHI agrees with agronomists, or its tiers are calibrated | No expert ratings yet (`docs/annotation/OHI_EXPERT_RATING.md`) |
| The forecaster predicts outbreaks, or has a lead time or false-alarm rate | No outbreak records; weather-only back-test |
| Results for ConvNeXt-T, Swin-T, YOLOv8s, RT-DETR, multi-seed ablations, 288 px | Not run (PROGRESS.md, *Deferred, not run*) |
| GPU or production-server latency; latency including the weather API | Only in-process CPU latency was measured |
| A working Docker deployment | The image was not built (`results/docker/build.txt`) |
| Leaf Spot detection | Not a model class |
