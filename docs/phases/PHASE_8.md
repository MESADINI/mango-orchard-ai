# Phase 8: Final audit

**Branch:** `enhance/real-data` · **Date:** 2026-10-17

## Checks
| Check | Result | Evidence |
|---|---|---|
| GPU available and selected | RTX 3050 4 GB, torch 2.3.0+cu121, `torch.cuda.is_available()` True. Disease inference selects `cuda`, pest inference selects device `0`. Both fall back to CPU when no device is visible. | `ml/disease_detection/inference.py::select_device`, `ml/pest_detection/inference.py::select_yolo_device` |
| Device of existing results | Every training and evaluation run logged `cuda_available: true` on the RTX 3050. Pest runs used `device: '0'`. Both latency benchmarks are CPU by design. Nothing was unfinished, so nothing was re-run. | `results/**/env.json`, `results/pest/*/train/args.yaml` |
| Lock file | Already records `torch==2.3.0+cu121` / `torchvision==0.18.0+cu121`; 143/143 entries match the venv. The install note wrongly said `--index-url`; corrected to `--extra-index-url`, which resolves (dry run). README now documents it. | `requirements-ml.lock`, `results/tests/phase8_reproduction.md` |
| Fresh clone | 231 passed, 12 skipped (data/weights git-ignored). Generated tables are byte-identical. Registry reports `missing` (exit 1). `npm ci` and the build pass. | `results/tests/phase8_reproduction.md` |
| Full suite, main repo | **243 passed**, 0 skipped (data, weights and GPU present) | `results/tests/phase8_pytest.txt` |
| Claims audit | Every README and paper claim mapped to evidence in `CLAIMS_EVIDENCE.md`. | see below |

## Changes from the claims audit
- **Paper:**
  - removed the names of planned-but-unrun backbones from the Discussion (per the Phase 2 instruction);
  - "fully reproducible" became "reproducible", because retraining was not repeated;
  - added the missing citation for the dataset authors' "about 1,800 distinct leaves".
- **README:** "the Docker image already includes" Pango/Cairo became "the Dockerfile installs them (image build not
  yet verified)".
- **`CLAIMS_EVIDENCE.md`:** rewritten as a full map (data, classifier, pest, XAI, OHI, weather, system), plus a
  "not supported" list.

## Final report
See the final summary in the session and `PROGRESS.md`. The publishability assessment:

**What is strong**
- Leakage-controlled data with grouping verified by hand.
- A pre-registered model selection rule.
- An honest negative external result: macro-F1 0.119 against 1.000 in-distribution.
- Calibrated abstention that catches 96.9% of out-of-domain photos.
- Explicit host-verification for proxy pest data.
- An OHI that never treats missing inputs as healthy, with a sensitivity analysis.
- A weather back-test that falsified the old alert rules.
- Every number generated from `results/`, with references checked by machine.

**What limits it**
- No field images from Indian orchards.
- No expert masks or ratings.
- Pest data from no mango host.
- Single runs, so no seed variance.
- A small external set (n = 381, 4 classes).
- No outbreak records.
- Docker untested.
- Paper not compiled and without figures.

**Rating: 5/10 as it stands.**
- Suitable for a workshop, a regional IEEE conference, or a negative-results / reproducibility track, framed as an
  audit ("benchmark accuracy is not field readiness").
- Not ready for a journal or a top venue, where a field-validation set or expert agreement study is the minimum.
- Collecting ≥ 50 expert masks and ≥ 20 OHI expert ratings, plus a small field image set, would move it to about
  7/10.
