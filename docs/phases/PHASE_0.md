# Phase 0 — Baseline and cleanup

**Branch:** `enhance/real-data` (created from `fix/verification-and-security` @ `1ec8837`)
**Date:** 2026-10-17

## Environment (evidence: `results/environment/20261017T065604Z.json`)

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200), AMD CPU, 16 logical cores |
| Python | 3.11.9 (`backend/venv`) |
| GPU (nvidia-smi) | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 617.42 |
| torch | 2.3.0 **CPU build** → `torch.cuda.is_available() == False` |
| make | not installed → `tasks.py` is the cross-platform runner; `Makefile` delegates to it |
| Docker | 29.7.2 |
| Kaggle credentials | **missing** (`~/.kaggle/kaggle.json` absent) |
| `ROBOFLOW_API_KEY` | **not set** |

GPU note: training on the GPU needs a CUDA wheel of torch, e.g.
`pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121`.
A 4 GB laptop GPU limits batch sizes for EfficientNet-B3 / ConvNeXt-T / Swin-T (mixed precision helps).
This is decided in Phase 2.

## Tests

- Before any change: `41 passed`.
- After Phase 0: `45 passed` (4 new tests in `ml/utils/tests/test_experiment.py`).
- Log: `results/tests/phase0_pytest.txt`.

## Unsupported claims removed

| File | Change |
|---|---|
| `CONFERENCE_PAPER_SUMMARY.md` | Marked **UNVERIFIED DRAFT**. Removed 98.4% accuracy, 91.2% mAP, the baseline table, per-class tables (incl. Leaf Spot, which has no data), the OHI sensitivity table, the Dharmapuri field trial / 34.2% reduction, "sub-2.5 s" latency and the unsourced production/yield-loss figures. Added notes where the paper's OHI formulas differ from `ml/severity/ohi.py` (γ term, f_RH/f_T/f_P, time window). Fixed stale architecture lines (SlowAPI, Supabase, Chart.js). |
| `PROJECT_PRESENTATION_SUMMARY.md` | Marked **UNVERIFIED DRAFT**. Removed demo numbers 89.2% / 82.5 / 14.5% (identical to the hard-coded stub in the initial commit), yield/pesticide impact percentages, the ">98% accuracy" Q&A answer, the 45% / 30–40% / 70% statistics, "72 hours in advance" and "fluent Tamil" claims. |
| `PROJECT_STATUS_ASSESSMENT.md` | Rewritten (v2.0.0): the earlier version claimed trained weights and 92–95% completion. |
| `PROJECT_PLAN.md` | Corrected phase statuses (data never downloaded, models never trained, comparison study not done). |

Check command (only hits are inside the removal notes themselves):
`grep -rn -E "98\.4|91\.2|34\.2|89\.2|82\.5" --include=*.md .`

## Added

- `ml/utils/experiment.py`: `set_seed`, `collect_env` (git commit/dirty, package versions, hardware incl. nvidia-smi), `start_experiment(exp_id, config)` → `results/<exp_id>/{config,env}.json`.
- `tasks.py` + `Makefile`: targets `env, data, train-disease, train-pest, eval, test, all`. They currently point at the existing scripts and will be updated as phases replace them.
- `requirements-ml.txt` (direct pins) and `requirements-ml.lock` (`pip freeze`, 141 packages). Added ImageHash 4.3.2 and kaggle 2.2.4.

## Exit criteria

- [x] Tests pass (45/45).
- [x] No unsupported claims remain in the three summary docs (plus `PROJECT_PLAN.md`).
