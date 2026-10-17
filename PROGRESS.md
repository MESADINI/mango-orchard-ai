# PROGRESS — `enhance/real-data`

Read this first when resuming. Every number in docs/paper must come from a file under `results/`.

| Phase | Title | Status | Evidence | Blockers |
|---|---|---|---|---|
| 0 | Baseline & cleanup | ✅ Done | `docs/phases/PHASE_0.md`, `results/environment/`, `results/tests/phase0_pytest.txt` | — |
| 1 | Real datasets | ✅ Done | `docs/phases/PHASE_1.md`, `docs/datasets/`, `results/data_stats.json`, `results/data/`, `results/tests/phase1_pytest.txt` | — (pest data is flagged non-mango proxy data) |
| 2 | Disease classifier training | ⏳ Next | — | 4 GB GPU limits batch size (use AMP, small batches) |
| 3 | Pest detector | Pending | — | Data exists but is proxy (trap/grape/coffee); must carry `host_verified: false` |
| 4 | Severity & explainability | Pending | — | Needs ≥50 lesion masks |
| 5 | OHI & forecaster | Pending | — | — |
| 6 | Integration & product quality | Pending | — | — |
| 7 | Documentation & paper | Pending | — | — |
| 8 | Final audit | Pending | — | — |

## Environment facts
- Windows 11, Python 3.11.9 in `backend/venv`, no `make` (use `python tasks.py <target>`).
- GPU: RTX 3050 Laptop 4 GB; torch 2.3.0+cu121 installed in Phase 1 (`torch.cuda.is_available() == True`).
- Data: `python tasks.py data` (needs `ROBOFLOW_API_KEY` in the environment; never print it). Raw/processed data are git-ignored.
- Disease split facts for Phase 2: 2,486 images / 1,085 leaves; test = 239 images / 159 leaves (small) -> 5-fold group CV is primary; `cv_fold` -2 = always train, -1 = test.

## Log
- 2026-10-17 — Phase 0 complete: branch created, 41 → 45 tests passing, invented claims removed, Makefile/tasks.py, ML lock file, experiment logger.
- 2026-10-17 — Phase 1 complete: MangoLeafBD verified against Mendeley SHA-256; near-duplicates grouped by DINOv2 retrieval + one-to-one SIFT/RANSAC (1,783 leaves vs authors' ~1,800); 5-class (Leaf Spot dropped); external set 381 images (972 MangoLeafBD-derived images found and excluded, no Healthy); pest set built as flagged proxy data; 90 tests passing.
