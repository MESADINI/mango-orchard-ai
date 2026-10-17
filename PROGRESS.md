# PROGRESS — `enhance/real-data`

Read this first when resuming. Every number in docs/paper must come from a file under `results/`.

| Phase | Title | Status | Evidence | Blockers |
|---|---|---|---|---|
| 0 | Baseline & cleanup | ✅ Done | `docs/phases/PHASE_0.md`, `results/environment/`, `results/tests/phase0_pytest.txt` | — |
| 1 | Real datasets | ✅ Done | `docs/phases/PHASE_1.md`, `docs/datasets/`, `results/data_stats.json`, `results/data/`, `results/tests/phase1_pytest.txt` | — (pest data is flagged non-mango proxy data) |
| 2 | Disease classifier training | ✅ Done (reduced scope) | `docs/phases/PHASE_2.md`, `docs/models/disease_classifier.md`, `results/disease/`, `results/tests/phase2_pytest.txt` | External-set macro-F1 0.119: does not transfer (see PHASE_2 §4) |
| 3 | Pest detector | 🔄 In progress (GPU training YOLOv8n / YOLO11n / YOLOv8s) | `results/pest/` | Proxy data (trap/grape/coffee): every class `host_verified: false` |
| 4 | Severity & explainability | 🔄 Partial: segmenter fixed, tooling + Grad-CAM deletion/insertion done | `docs/phases/PHASE_4.md`, `results/xai/` | **Blocked on user:** ≥50 expert lesion masks (`docs/annotation/SEVERITY_MASKS.md`) for severity validation and the pointing game |
| 5 | OHI & forecaster | 🔄 Done in working tree, not yet committed | `docs/phases/PHASE_5.md`, `results/ohi/`, `results/forecaster/` | Expert OHI ratings needed for validation (`docs/annotation/OHI_EXPERT_RATING.md`); Open-Meteo archive unreachable here, NASA POWER used |
| 6 | Integration & product quality | 🔄 In progress: registry, citations, chatbot, e2e tests done | `docs/phases/PHASE_6.md`, `docs/sources.md`, `ml/model_registry.json` | API latency (after GPU work), Docker build (daemon not running) |
| 7 | Documentation & paper | 🔄 Skeleton: results generator, ARCHITECTURE.md | `docs/RESULTS.md`, `docs/ARCHITECTURE.md`, `paper/generated/numbers.tex` | Waits for Phases 3–6 results |
| 8 | Final audit | Pending | — | — |

## Environment facts
- Windows 11, Python 3.11.9 in `backend/venv`, no `make` (use `python tasks.py <target>`).
- GPU: RTX 3050 Laptop 4 GB; torch 2.3.0+cu121 installed in Phase 1 (`torch.cuda.is_available() == True`).
- Data: `python tasks.py data` (needs `ROBOFLOW_API_KEY` in the environment; never print it). Raw/processed data are git-ignored.
- Disease split facts for Phase 2: 2,486 images / 1,085 leaves; test = 239 images / 159 leaves (small) -> 5-fold group CV is primary; `cv_fold` -2 = always train, -1 = test.

## Deferred, not run
These runs were planned but **not run**. No numbers exist for them; they must not appear in the paper, README or
CLAIMS_EVIDENCE.md. The code still supports them (`--archs convnext_tiny,swin_tiny`).
- Phase 2, ConvNeXt-T and Swin-T 5-fold CV and final models: too slow on the 4 GB GPU for the time budget. One
  ConvNeXt-T fold-0 run was stopped at epoch 5 and its partial output was removed from `results/`.
- Phase 2, the original 15 ablation runs (5 variants × 3 seeds on the train/val/test split: baseline,
  no pretraining, basic augmentation, no label smoothing + no class weights, leaky image-level random split).
  Replaced by 3 single runs on CV fold 0 (pretrained vs scratch, strong vs basic augmentation, 224 vs 256 px),
  each compared with the existing EfficientNet-B3 CV fold-0 run. No seed variance is estimated.
- Phase 2, 288 px input size: does not fit in 4 GB at batch 32 (measured peak 4.52 GB); 256 px used instead
  (3.63 GB).

## Log
- 2026-10-17 — Phase 0 complete: branch created, 41 → 45 tests passing, invented claims removed, Makefile/tasks.py, ML lock file, experiment logger.
- 2026-10-17 — Phase 1 complete: MangoLeafBD verified against Mendeley SHA-256; near-duplicates grouped by DINOv2 retrieval + one-to-one SIFT/RANSAC (1,783 leaves vs authors' ~1,800); 5-class (Leaf Spot dropped); external set 381 images (972 MangoLeafBD-derived images found and excluded, no Healthy); pest set built as flagged proxy data; 90 tests passing.
- 2026-10-17 — Phase 2 scope reduced on request: CV done for EfficientNet-B3, ResNet-50, MobileNetV3-L (15/15 runs); ConvNeXt-T/Swin-T and 12 of 15 ablations deferred, not run; final models + 3 fold-0 ablations running. Non-GPU work for Phases 4–6 started in parallel.
- 2026-10-17 — Phase 2 complete: EfficientNet-B3 selected (CV macro-F1 0.998 ± 0.004; no significant difference to ResNet-50 / MobileNetV3-L); test 1.000 but external macro-F1 0.119; low-confidence flag on 96.9% of external images; fp32 re-evaluation of all 21 runs; T = 1 deployed (fit not identifiable, post-hoc rule disclosed). Deployed SHA c6becbe0…8ff1.
- 2026-10-17 — Non-GPU work while the GPU trains the pest detectors:
  - OHI redesigned: missing components excluded, no image component → no OHI, off-host and unverified pests excluded.
  - Sensitivity analysis: ranking robust, but tiers change for up to 12.1% of scenarios at ±20% weights and 42.3% (median) with uniform weights.
  - Forecaster back-test on 10 years of NASA POWER data: the documented 'High' conditions never occur; the previous rules flagged 216.2 days/year.
  - Citations re-verified (banned and unsupported chemicals removed, see docs/sources.md); chatbot fixed; model registry and /api/models added.
  - CPU-device bug in select_device fixed; frontend dependencies trimmed; results tables generated from results/.
