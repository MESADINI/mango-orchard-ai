# AI Mango Orchard Health & Pest Intelligence — Project Status Assessment

**Document Version:** 2.0.0 (replaces v1.0.0)
**Assessment Date:** 2026-10-17
**Repository:** `mango-orchard-ai`, branch `enhance/real-data`

> **Why this document was rewritten.** Version 1.0.0 stated that the models were trained ("weights trained and
> saved at `ml/disease_detection/weights/efficientnet_b3_best.pth`"), that the project was "92–95% complete",
> and that a manuscript was "ready". An audit on 2026-10-17 found **no trained weights, no training logs, no
> processed datasets and no result files** anywhere in the repository or its git history. Those statements
> were not true and have been removed. Live, evidence-linked status is tracked in [`PROGRESS.md`](PROGRESS.md).

---

## 1. Summary

| Area | Status | Evidence |
|---|---|---|
| Web platform (FastAPI backend, React frontend, SQLite, JWT auth, PDF/HTML reports) | Built; runs | `backend/`, `frontend/`, `backend/tests/test_api.py` |
| Test suite | 41 tests pass (software behaviour only; models are random-weight/stubbed) | `docs/phases/PHASE_0.md` |
| Disease classifier (EfficientNet-B3) | Code only — **never trained** | No `.pth` file exists |
| Pest detector (YOLOv8) | Code only — **never trained**; no script merges the real datasets | No `.pt` file exists |
| Real datasets | **Not downloaded**; only 12 synthetic cartoon images are in the repo | `ml/datasets/sample_*` |
| Leaf Spot class | No data source (absent from MangoLeafBD) | `ml/datasets/README.md` |
| Grad-CAM | Implemented; tested on random weights only; not quantitatively evaluated | `ml/explainability/` |
| HSV lesion severity | Implemented; not validated against masks | `ml/severity/severity.py` |
| OHI engine | Implemented; constants hand-chosen; not validated; paper formulas differ from code | `ml/severity/ohi.py` |
| Weather risk forecaster | Implemented heuristic; not validated against outbreak data | `ml/forecasting/forecaster.py` |
| Treatment citations | Present, but point to portal home pages and have not been re-verified | `ml/forecasting/recommendations_db.json` |
| Tamil chatbot | Keyword-based; Tamil covers Anthracnose + greeting only | `backend/app/api/chatbot.py` |
| Experimental results for a paper | **None** | — |

## 2. What works today

- Upload → validation → pipeline → stored scan → report, with honest "unavailable" reporting and a PARTIAL OHI
  when models or live weather are missing.
- Rule 1 (deterministic OHI), Rule 3 (UNVERIFIED flag for Mango Hopper / Stem Borer) and Rule 4 (disclosed nutrient
  placeholder) are implemented in code. Rule 2 (citations) is implemented structurally, but the citations themselves
  need verification.

## 3. What remains (see `PROGRESS.md` for the phased plan)

1. Download and deduplicate real datasets; resolve the Leaf Spot class; build the merged pest dataset.
2. Train and evaluate the disease classifier and baselines with cross-validation; train the pest detector.
3. Validate lesion severity, Grad-CAM, the OHI and the weather rules.
4. Verify every treatment citation against live sources.
5. Rewrite the paper using only measured results.

## 4. Known issues found in the audit

- Missing disease model makes the OHI treat the leaf as "Healthy" (optimistic score; flagged as partial).
- UNVERIFIED pest detections are still included in the OHI pest penalty.
- HSV segmenter likely misses black, low-saturation sooty mould.
- YOLO training copies the first `best.pt` found by a glob across all runs, and uses cwd-relative paths.
- `MANUAL_ANNOTATION_GUIDE.md` references a non-existent `--include-all` flag.
- `chart.js` and the frontend Supabase client are unused.
