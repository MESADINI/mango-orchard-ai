# Phase 6: Integration and product quality

**Branch:** `enhance/real-data` · **Date:** 2026-10-17

## 1. Model registry and uncertainty UI
- **Registry builder.** `ml/model_registry.py build` writes `ml/model_registry.json`, one entry per deployed
  model, containing:
  - version (first 12 hex digits of the SHA-256) and the full hash;
  - path and class list;
  - calibration: temperature and low-confidence threshold;
  - pest `host_verified` flags;
  - source run, selection rule, and headline metrics copied from `results/`.
- **Verification.** `verify` exits non-zero if the installed weights do not match. `GET /api/models` reports
  `ok` / `missing` / `sha_mismatch` per model.
- **Uncertainty UI.** Below the calibrated threshold, the result page shows a "retake photo" banner. The disease
  result is excluded from the OHI, and no treatment is suggested.
- **Device bug fixed.** `select_device` chose CUDA whenever `torch.cuda.is_available()` was true, even with no
  visible device (for example `CUDA_VISIBLE_DEVICES=""`). Loading the checkpoint then crashed. A visible device is
  now required. Regression test: `test_select_device_needs_a_visible_cuda_device`.

## 2. Treatment citations (`docs/sources.md`)
- Every entry was checked against the live TNAU pages, the NHM IPM bulletin and the CIB&RC banned / restricted
  list (updated 31.07.2026). Each entry now carries a deep link, a verbatim quote and the verification date.
- **Removed:**
  - malathion: banned on mango;
  - dimethoate: banned on fruit eaten raw;
  - dichlorvos and phosphamidon: banned;
  - dinocap and tridemorph: banned;
  - azoxystrobin and thiamethoxam: sources not found;
  - Streptocycline: draft phase-out.

  Several doses and timings were corrected to what the sources actually say.
- **Pre-harvest interval.** It is stated only where the source gives one: carbendazim, 15 days.
- **Not verified:** product-by-product label claims for mango. The app tells the user to check the label.
- **SQL seed.** `database/seeds/001_seed_recommendations.sql` is generated from the JSON by
  `database/seeds/generate_seed.py`. A test keeps the two in sync.

## 3. Chatbot (`backend/app/api/chatbot.py`)
- **English matching:** word-boundary and hyphen-aware, so "large-scale" does not match "scale"; plurals are
  handled.
- **Tamil matching:** keywords match at the start of a token (Tamil suffixes attach to the stem). The disease and
  pest names now have Tamil keywords.
- **Safe fallback** in both languages when nothing matches: it points to the local KVK or agriculture officer and
  never guesses.
- **Notices:** unverified-pest notices in both languages, and a label caveat.
- **Limits:** at most 3 conditions per answer. Responses include `source_urls`.
- **Tests:** `backend/tests/test_chatbot.py`.

## 4. End-to-end, latency, Docker
- **End-to-end tests with the real weights:** `ml/tests/test_e2e_real_models.py`, CPU only, skipped when the
  weights are absent. They cover:
  - registry match;
  - calibration embedded in the checkpoint;
  - one test image per class through the full pipeline;
  - the full response with Grad-CAM;
  - the pest detector flagged host-unverified.
- **API latency:** `backend/benchmark_latency.py` → `results/api/latency.{json,md}`.
  **Results** (CPU, in-process client, weather stubbed, 30 requests after 3 warm-up, commit 649aa60, from
  `results/api/latency.md`):

  | Scan type | Median (ms) | p95 (ms) |
  |---|---|---|
  | disease | 345 | 403 |
  | pest | 99 | 105 |
  | dual | 404 | 446 |

  The dual scan includes Grad-CAM, severity segmentation, OHI and the database write. The network weather call is
  excluded.
- **Docker build:** not run. The Docker daemon was not running in this environment (attempt recorded in
  `results/docker/build.txt`). Building the image remains an open check.
