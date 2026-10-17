# Phase 8 fresh-clone reproduction

- **Clone:** `git clone -b enhance/real-data` of commit `dbf2956` (417 tracked files) into an empty directory.
- **Environment:** Python 3.11.9, torch 2.3.0+cu121, RTX 3050 4 GB, Windows 11.
- **Driver script:** `phase8_repro.sh` (session scratchpad). Commands are listed below.
- **Packages:** all 143 entries of `requirements-ml.lock` match the venv's `pip freeze` (0 missing or different).
- **CUDA wheels:** `torch==2.3.0+cu121` and `torchvision==0.18.0+cu121` resolve with
  `--extra-index-url https://download.pytorch.org/whl/cu121` (`pip install --dry-run`).

| Check | Command | Result |
|---|---|---|
| No weights or secrets tracked | `git ls-files \| grep -E '\.(pt\|pth\|onnx\|ckpt)$\|\.env$'` | 0 files |
| Test suite (no data, no weights) | `python -m pytest ml backend/tests -q -rs` | **231 passed, 12 skipped**, exit 0. All 12 skips: git-ignored datasets (8) or weights (4) |
| Generated tables are reproducible | `python tasks.py paper-tables`, then `git status --porcelain` | **0 changed files** (README block, `docs/RESULTS.md`, `paper/generated/*.tex` byte-identical) |
| Registry without weights | `python ml/model_registry.py verify` | exit 1, both models `"status": "missing"` (expected: weights are not committed) |
| Registry in the main repo | same | exit 0, both SHA-256 match |
| Frontend clean install | `npm ci && npm run build` (frontend/) | exit 0, built in 1.5 s |

Not reproduced: retraining the models. Training reproducibility rests on the logged seeds, configs and environments
in `results/disease/**/env.json` and `results/pest/*/env.json`, which show all runs on the RTX 3050. Retraining was
not repeated, so bit-exactness is not claimed. Full suite with data, weights and GPU: 243 passed (`phase8_pytest.txt`).
