# Phase 7: Documentation and paper

**Branch:** `enhance/real-data` · **Date:** 2026-10-17

## Generated, not typed
`python tasks.py paper-tables` (`ml/reporting/results_tables.py`) reads only `results/` and writes four outputs:

| Output | Content |
|---|---|
| `docs/RESULTS.md` | every generated table, each with its source file |
| README results block | the table between `<!-- RESULTS:START/END -->` |
| `paper/generated/numbers.tex` | LaTeX macros for every number quoted in the paper text |
| `paper/generated/tables.tex` | the paper's tables: CV, final models, ablation, pest, Grad-CAM, OHI |

A result that does not exist yet renders as "pending (`results/...`)" or `\pending{...}`, never as a number. Tests
(`ml/reporting/tests/`) check this and the README block.

## Documents
- **`README.md`:** rewritten.
  - The intro states what the system does and how far each part is validated.
  - The headline results are generated.
  - The known limitations are taken from `results/`.
  - Training instructions match `tasks.py`.
- **`docs/ARCHITECTURE.md`:**
  - components and the request flow;
  - when each step is skipped;
  - the provenance chain from training run to registry to API.
- **Model cards:** `docs/models/disease_classifier.md` and `docs/models/pest_detector.md`. Data cards from Phase 1
  are in `docs/datasets/`.
- **`CLAIMS_EVIDENCE.md`:** every allowed claim with its evidence file, and the claims that must not be made.
- **`paper/main.tex`** (IEEEtran):
  - framed around the evidence: in-distribution near-ceiling accuracy, failure on independent images, calibrated
    abstention, proxy pest data, OHI sensitivity, weather back-test;
  - results come only from the generated macros and tables;
  - the dataset counts in the text are copied from `results/data_stats.json` and PHASE_1.

  `CONFERENCE_PAPER_SUMMARY.md` is marked superseded.

## References
`paper/refs.bib` (26 entries) is checked by `ml/reporting/verify_refs.py` against Crossref (DOI) and arXiv (eprint)
→ `results/paper/references_check.json`. Result:
- 21 entries match on title and year (±1 year allowed for preprints);
- 5 are not machine-verifiable: Ultralytics software, Holm (1979, JSTOR), NHM bulletin, CIB&RC list, TNAU portal.
  The URLs of the last three were opened during the citation audit (`docs/sources.md`).

## Not done
- **PDF build.** The manuscript is not compiled: no TeX distribution is installed in this environment. Compile with
  `pdflatex main && bibtex main && pdflatex main && pdflatex main` in `paper/`, or on Overleaf. The
  `generated/*.tex` files must be regenerated first.
- **Figures.** The paper has no figures yet. Candidates are in `results/disease/figures/`, `results/xai/curves.png`,
  `results/ohi/oat.png` and `results/forecaster/monthly.png`.

## Fix found while testing
With `CUDA_VISIBLE_DEVICES=""`, Ultralytics' own device auto-selection asserted on a CUDA build of torch with no
visible device, so pest inference crashed on CPU-only hosts. This is the same class of bug as the Phase 6
`select_device` fix. `PestDetector` now takes a `device`, chosen by `select_yolo_device` (requires a visible GPU), and
the pipeline passes it through. A regression test is in `ml/pest_detection/tests/test_pest_detection.py`. The full
suite (`ml`, `backend/tests`) passes on CPU: 243 tests.
