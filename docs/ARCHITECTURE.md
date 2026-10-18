# Architecture

This page shows how a photo becomes an Orchard Health Index and a set of cited treatments. Every box below names
the file that implements it. Measured results are in [RESULTS.md](RESULTS.md), which is generated from
`results/`.

## 1. Components

```
React (frontend/src)                 FastAPI (backend/app)                       ML package (ml/)
 Upload.jsx ── POST /api/analyze ──▶ api/analyze.py ──▶ services/pipeline_service.py ──▶ pipeline.py
 ScanResult.jsx ◀── AnalysisResponse ─┘      │                                      │
 Dashboard / History ── GET /api/analyze     ├─ core/store.py (SQLite: users, farms, trees, scans)
 Chatbot.jsx ── POST /api/chatbot ──▶ api/chatbot.py ──▶ forecasting/recommendations_db.json
 Dashboard ── GET /api/weather ──▶ api/weather.py ──▶ forecasting/weather.py + forecaster.py
 report link ── GET /api/report ──▶ api/report.py (PDF)
 /api/models ──▶ main.py ──▶ model_registry.py ──▶ ml/model_registry.json
```

## 2. One scan (`ml/pipeline.py`, `MangoOrchardPipeline.analyze_orchard`)

| Step | Module | What happens | When it is skipped or marked unavailable |
|---|---|---|---|
| 1. Disease | `disease_detection/inference.py` | Deployed timm classifier (registry entry `disease_classifier`); softmax at the deployed temperature | No weights → "unavailable". Top probability below the calibrated threshold → `low_confidence`, "retake photo", disease excluded from the OHI |
| 2. Explanation | `explainability/gradcam.py` | Hook-based Grad-CAM on `conv_head` → overlay image | Model unavailable, or `generate_gradcam=False` |
| 3. Severity | `severity/severity.py` | HSV leaf/lesion segmentation. Sooty mould = dark relative to the leaf; soil and shadow are excluded by keeping the largest filled green component | No leaf found → 0% lesion area, with the leaf pixel count reported |
| 4. Pests | `pest_detection/inference.py` | Deployed YOLO detector (registry entry `pest_detector`). Each box carries `is_unreliable` (Mango Hopper / Stem Borer, < 100 samples) and `host_verified` (class trained on mango images?) | No weights → no detections. No host-verified class → the pest component is missing |
| 5. Weather | `forecasting/weather.py`, `forecaster.py` | Open-Meteo current, daily and hourly data. Each day gets a disease-favourable-weather level from the NHM bulletin thresholds | Unreachable → weather component missing, forecast "Unknown". Never a default value |
| 6. OHI | `severity/ohi.py` | Weighted penalties. Missing components are excluded and the remaining weights renormalised. No image component → no OHI. See [OHI_METHODOLOGY.md](OHI_METHODOLOGY.md) | — |
| 7. Treatments | `forecasting/recommendations.py` | Cited TNAU / NHM protocols (deep links, quotes, verification date) for confident, verified detections only | Uncertain disease, or unverified / off-host pests → none |

The response lists every missing component and caveat (`missing_components`, `is_partial`, `caveats`). The
frontend shows "not assessed" / "—" for them.

## 3. Models and provenance
- **Training:**
  - disease: `ml/disease_detection/experiments.py`, protocol in `docs/phases/PHASE_2.md`;
  - pest: `ml/pest_detection/train.py`, protocol in `docs/phases/PHASE_3.md`.

  Every run writes `config.json`, `env.json` (git commit, package versions, hardware, seed) and `summary.json`
  with the checkpoint SHA-256 under `results/`.
- **Deployment:** each `deploy` step copies the selected checkpoint by explicit path and verifies its SHA-256.
  `ml/model_registry.py build` records version, hash, classes, calibration and the host flags in
  `ml/model_registry.json`. `/api/models` reports whether the installed weights match.
- **Weights** are not committed. Datasets are rebuilt with `python tasks.py data`, using verified downloads.

## 4. Data flow of evidence
`results/**` (written only by code) → `ml/reporting/results_tables.py` (`python tasks.py paper-tables`) →
`docs/RESULTS.md` and the README table. Documents may quote a number only if it appears in one of those files.

## 5. Deployment
`docker-compose.yml` runs two services:
- the backend, uvicorn on :8000, with `ml/` mounted for the code, weights and recommendation database,
  and SQLite and media in volumes;
- the frontend, nginx on :3000.

Configuration comes from `backend/.env` (see `backend/.env.example`). No external key is needed: Open-Meteo is
keyless, and authentication is local JWT with bcrypt.
