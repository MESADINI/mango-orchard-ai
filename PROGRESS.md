# PROGRESS — `enhance/real-data`

Read this first when resuming. Every number in docs/paper must come from a file under `results/`.

| Phase | Title | Status | Evidence | Blockers |
|---|---|---|---|---|
| 0 | Baseline & cleanup | ✅ Done | `docs/phases/PHASE_0.md`, `results/environment/`, `results/tests/phase0_pytest.txt` | — |
| 1 | Real datasets | ⏳ Next | — | Kaggle credentials missing; `ROBOFLOW_API_KEY` not set |
| 2 | Disease classifier training | Pending | — | torch is a CPU build (RTX 3050 4 GB present) |
| 3 | Pest detector | Pending | — | Needs Phase 1 pest data |
| 4 | Severity & explainability | Pending | — | Needs ≥50 lesion masks |
| 5 | OHI & forecaster | Pending | — | — |
| 6 | Integration & product quality | Pending | — | — |
| 7 | Documentation & paper | Pending | — | — |
| 8 | Final audit | Pending | — | — |

## Environment facts
- Windows 11, Python 3.11.9 in `backend/venv`, no `make` (use `python tasks.py <target>`).
- GPU: RTX 3050 Laptop 4 GB, but torch 2.3.0 is a CPU wheel.

## Log
- 2026-10-17 — Phase 0 complete: branch created, 41 → 45 tests passing, invented claims removed, Makefile/tasks.py, ML lock file, experiment logger.
