# Claims and evidence

Every claim in `README.md` and `paper/main.tex`, each with the file that supports it.
- Numeric claims point to a file under `results/` written by code. Regenerate the tables and LaTeX macros with
  `python tasks.py paper-tables`.
- Behavioural claims ("never", "excluded", "gated") point to the code and to the tests that enforce it. The tests
  pass in `results/tests/phase8_pytest.txt`.
- A statement that appears in no table below must not be made. Audited 2026-10-17 (Phase 8); claims without
  evidence were removed from the README and the paper.

## Data
| # | Claim (README / paper) | Evidence |
|---|---|---|
| D1 | MangoLeafBD has 4,000 images; the dataset authors state about 1,800 are distinct leaves | `results/data_stats.json` (`mangoleafbd_download.n_images`); authors' statement cited (`ahmed2023mangoleafbd`); 1,783 groups found (`docs/phases/PHASE_1.md`) |
| D2 | Five classes are used; Leaf Spot is not a model class | `results/data_stats.json` (`disease.classes`, `class_map`) |
| D3 | Copies of a leaf are grouped (DINOv2 retrieval, SIFT + RANSAC verification); no group crosses splits | `results/data_stats.json` (`disease.dedup`), `results/data/disease_dedup_calibration.json`, `ml/datasets/tests/test_data_integrity.py` |
| D4 | 2,486 images from 1,085 leaves; test split of 239 images | `results/data_stats.json` (`disease.n_images`, `n_groups`), `results/disease/deployed_eval.json` |
| D5 | External set: 381 independent images of 4 classes; 972 MangoLeafBD-derived images excluded | `results/data_stats.json` (`external_test`), `results/data/external_testset.json` |
| D6 | Pest data: 3 public sets, 2,559 images in 1,400 groups; none shows mango (trap / grape and stock / coffee) | `results/data/pest_dataset.json`, `results/data/pest_review/`, `ml/datasets/pest_source_review.json` |
| D7 | Data downloads verify SHA-256 checksums | `results/data/mangoleafbd_download.json` (`matched_to_mendeley_sha256`), `ml/datasets/` |

## Disease classifier
| # | Claim | Evidence |
|---|---|---|
| C1 | 5-fold group CV macro-F1 of the selected EfficientNet-B3 is 0.998 ± 0.004; selection rule fixed in advance | `results/disease/summary.json` (`cv`, `selection_rule`), `results/disease/tables/cv_comparison.md` |
| C2 | The three backbones are statistically indistinguishable (McNemar and Wilcoxon, Holm-corrected) | `results/disease/tables/significance.md` |
| C3 | Test-split accuracy 1.000 (n = 239, in-distribution) | `results/disease/tables/final_models.md`, `results/disease/deployed_eval.json` |
| C4 | **External macro-F1 0.119 [0.083, 0.155]** (n = 381): the model does not transfer | `results/disease/summary.json`, `results/disease/tables/external_selected.md` |
| C5 | The low-confidence rule flags 7.9% of test and 96.9% of external images ("retake photo") | `results/disease/summary.json` |
| C6 | OOD AUROC (near / far) and ECE as reported in the final-model table | `results/disease/tables/final_models.md`; `paper/generated/tables.tex` (tab:final) |
| C7 | Only removing ImageNet pretraining has a measurable ablation effect (fold-0 macro-F1 1.000 → 0.649, McNemar p = 3.5e-18); single runs | `results/disease/tables/ablation.md` |
| C8 | Deployed temperature is 1.0 because the validation fit is not identifiable (post-hoc rule, disclosed) | `results/disease/summary.json`, `results/disease/deployed_eval.json`, `ml/model_registry.json` |
| C9 | Evaluation runs in fp32; AMP autocast gave non-finite logits on OOD images | `docs/phases/PHASE_2.md` §"fp16 overflow" (an engineering observation, not a results/ metric) |
| C10 | Training ran on the RTX 3050 with seed 42 and a logged environment | `results/disease/{cv,final,ablation}/**/env.json`, `config.json`; `results/pest/*/env.json` |

## Pest detector
| # | Claim | Evidence |
|---|---|---|
| P1 | YOLO11n (deployed): test mAP50-95 0.520; YOLOv8n 0.519; selected on validation mAP50-95 | `results/pest/models.md`, `results/pest/deployed.json`, `results/pest/{yolov8n,yolo11n}/summary.json` |
| P2 | Mealy Bug is poor (AP50 0.26, AP50-95 0.12) | `results/pest/yolo11n/summary.json`, `results/pest/models.md` |
| P3 | COCO weights pinned by SHA-256; trained once, seed 42, deterministic | `results/pest/*/config.json`, `train/args.yaml`, `ml/pest_detection/train.py` |
| P4 | All classes `host_verified: false`; detections shown as unverified and excluded from the OHI | `ml/pest_detection/weights/pest_detector.json`, `ml/model_registry.json`, `ml/tests/test_e2e_real_models.py`, `ml/tests/test_full_pipeline.py` |
| P5 | Mango Hopper and Stem Borer are gated (fewer than 100 labelled examples) | `ml/pest_detection/inference.py`, `ml/pest_detection/tests/test_pest_detection.py` |

## Lesion area and explanations
| # | Claim | Evidence |
|---|---|---|
| X1 | The HSV segmenter handles sooty mould and soil backgrounds | `ml/severity/`, `ml/severity/tests/`, `docs/phases/PHASE_4.md` (behaviour on test cases; accuracy not validated, see "Not supported") |
| X2 | Grad-CAM beats a centre prior modestly: deletion AUC 0.606 vs 0.629, insertion 0.821 vs 0.807, better on ~60% of images (Wilcoxon p < 0.001) | `results/xai/faithfulness.md`, `faithfulness.json` |
| X3 | Random-order deletion collapses predictions faster than any saliency order | `results/xai/faithfulness.json` (random deletion AUC 0.286) |

## Orchard Health Index
| # | Claim | Evidence |
|---|---|---|
| O1 | Unassessed components are excluded and weights renormalised; with no image component there is no OHI | `ml/severity/ohi.py`, `ml/severity/tests/test_ohi.py`, `ml/tests/test_full_pipeline.py` |
| O2 | Missing inputs are disclosed (e.g. nutrient defaulted) | `ml/severity/ohi.py`, `ml/severity/tests/test_ohi.py` |
| O3 | Rankings are robust: Spearman ρ ≥ 0.99 for ±20% on one weight (239 scenarios) | `results/ohi/sensitivity.md`, `sensitivity.json` |
| O4 | Tiers are not robust: up to 12.1% change (one weight); 42.3% median with uniform Dirichlet weights | `results/ohi/sensitivity.md` |

## Weather rules
| # | Claim | Evidence |
|---|---|---|
| W1 | Thresholds are quoted from the NHM IPM bulletin | `results/forecaster/backtest.json` (`thresholds_source`), `docs/sources.md` |
| W2 | 2015–2024 NASA POWER, Krishnagiri: documented "High" conditions occur on 0.0 days/year for all three diseases | `results/forecaster/backtest.json` (`documented_conditions_days_per_year`), `monthly.md` |
| W3 | RH ≥ 95% is common (1,364 h/year) but never at 25–30 °C | `results/forecaster/backtest.json` (`weather_summary`) |
| W4 | The pre-audit rules raised High on 216.2 days/year | `results/forecaster/backtest.json` (`previous_rules_high_days_per_year`) |

## Treatments, system and reproducibility
| # | Claim | Evidence |
|---|---|---|
| S1 | Each treatment links to and quotes TNAU / NHM and is checked against the CIB&RC banned list (31.07.2026) | `docs/sources.md`, `ml/forecasting/recommendations_db.json`, `ml/forecasting/tests/test_recommendations.py` |
| S2 | The chatbot is keyword-based, bilingual (English/Tamil), and answers only from the verified database | `backend/app/api/chatbot.py`, `backend/tests/test_chatbot.py` |
| S3 | Deployed weights match the registry (SHA-256); without weights, results are "unavailable", never fabricated | `ml/model_registry.json`, `ml/tests/test_model_registry.py`, `results/tests/phase8_reproduction.md` |
| S4 | Median POST /api/analyze latency, CPU (16 threads), weather stubbed: dual 404 ms, disease 345 ms, pest 99 ms | `results/api/latency.md`, `latency.json` |
| S5 | Every reported number is produced by code; tables regenerate byte-identically in a fresh clone | `results/tests/phase8_reproduction.md` |
| S6 | 21 of 26 references verify against Crossref / arXiv; 5 are software, JSTOR or web sources checked by hand | `results/paper/references_check.json` |
| S7 | The full test suite passes with data, weights and GPU present | `results/tests/phase8_pytest.txt` |

## Not supported (do not claim)
| Claim | Why not |
|---|---|
| Field or orchard accuracy of the disease model | The only independent evaluation (C4) shows failure |
| The pest detector works on mango trees | No mango images in training or test |
| Lesion-area accuracy | No expert masks yet (`docs/annotation/SEVERITY_MASKS.md`) |
| Grad-CAM localises lesions | The pointing game needs the masks |
| The OHI agrees with agronomists, or its tiers are calibrated | No expert ratings yet (`docs/annotation/OHI_EXPERT_RATING.md`) |
| The forecaster predicts outbreaks, or has a lead time or false-alarm rate | No outbreak records; weather-only back-test |
| Results for any configuration not listed above | Not run (`PROGRESS.md`, *Deferred, not run*) |
| Seed variance over training runs | Every training configuration was run once |
| GPU or production-server latency; latency including the weather API | Only in-process CPU latency was measured |
| A working Docker deployment | The image was not built (`results/docker/build.txt`) |
| Bit-exact retraining from a fresh clone | Reproduction covers code, tests and generated tables; retraining was not repeated |
