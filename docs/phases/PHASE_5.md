# Phase 5: OHI correctness, sensitivity analysis, forecaster back-test

**Branch:** `enhance/real-data` · **Date:** 2026-10-17
**Exit criterion met:**
- `results/ohi/` (sensitivity analysis);
- `results/forecaster/` (back-test);
- the OHI documentation matches the code (a test pins its worked example);
- tests for every change.

**Not met:** expert validation. The scaffolding is ready, but no ratings exist yet.

All numbers below are copied from `results/ohi/sensitivity.md` and `results/forecaster/backtest.json`.

## 1. OHI fixes (`ml/severity/ohi.py`, `ml/pipeline.py`)
| Before | Now |
|---|---|
| A missing disease model scored the leaf as Healthy | Missing components are excluded and the other weights renormalised proportionally. With no image component there is no OHI (`ohi_score: null`, status "Unknown — insufficient data") |
| Offline weather fell back to fixed default values | Weather is excluded when live data is missing. No fabricated defaults |
| Uncertain predictions were scored as confident | A prediction below the calibrated low-confidence threshold excludes the disease component and shows a "retake photo" message |
| Healthy prediction with large lesion area went unnoticed | `healthy_lesion_inconsistency` is flagged at ≥ 5% lesion area, with a caveat |
| Mango Hopper / Stem Borer detections entered the pest penalty | Excluded, counted in `excluded_unverified_pest_detections` |
| Pest classes trained on other hosts were treated as mango evidence | `host_verified` flag per class (from `ml/datasets/pest_source_review.json`). Off-host detections are excluded. A detector with no host-verified class makes the pest component missing |
| An unknown class name silently scored 0 | It raises `ValueError` |
| Treatment advice was given even for uncertain diagnoses | Recommendations are issued only for confident, verified detections |

The `host_verified` plumbing was a Phase 3 requirement. It lands in this commit because it changes OHI semantics.
The detector training is in the Phase 3 commit.

The API, PDF report and frontend now show "not assessed" / "—" for missing values instead of numbers. The frontend
no longer depends on chart.js, react-chartjs-2 or @supabase/supabase-js, none of which were used; `npm run build`
passes. The `--include-all` reference in the annotation guide had already been corrected in Phase 1.

`docs/OHI_METHODOLOGY.md` has been rewritten to match the code:
- proportional renormalisation, replacing the "+0.05 / +0.05" text;
- no γ attenuation term;
- the exact weather curve, with no 2 mm threshold;
- the exclusion rules;
- a worked example (63.10), pinned by `test_methodology_worked_example`.

## 2. Sensitivity analysis (`ml/severity/ohi_sensitivity.py` → `results/ohi/`)
**Scenarios** (n = 239, the MangoLeafBD test split):
- the deployed classifier's saved fp32 test predictions, with the low-confidence rule applied;
- severity from the HSV segmenter;
- pest boxes from a randomly paired pest test image;
- one randomly drawn hour of 2015–2024 NASA POWER weather for Krishnagiri;
- nutrient missing.

The pairing is synthetic. It exercises the formula over realistic input ranges; it is not a field sample.

The nominal OHI has mean 61.1 (SD 18.7), range 14.2–96.6.

| Perturbation | Result |
|---|---|
| One weight × 0.8–1.2 | Spearman ρ ≥ 0.993; up to 12.1% of tiers change |
| One α or β × 0.8 / 1.2 | ρ ≥ 0.994; up to 7.5% of tiers change |
| Dirichlet weights, c = 100 / c = 20 | median ρ 0.996 / 0.983; median tier change 7.5% / 17.2% |
| Uniform Dirichlet (no prior) | median ρ 0.831; median tier change 42.3% |
| Remove disease / pest / weather | tiers change 48.1% / 56.4% / 18.8% |

**Conclusion.** The ordering of scenarios is robust to moderate weight changes. The tier labels are not: they
depend on the expert-chosen weights. Expert validation is therefore required before tiers are presented as
agronomic advice. Scaffolding:
- `ml/severity/expert_agreement.py`, a blind 40-scenario sheet stratified by tier
  (`ml/datasets/annotation/ohi_expert_ratings_template.csv`);
- Spearman agreement with a bootstrap CI, plus inter-rater agreement;
- instructions in `docs/annotation/OHI_EXPERT_RATING.md`.

## 3. Forecaster (`ml/forecasting/forecaster.py`, `backtest.py` → `results/forecaster/`)
The rules now use the thresholds of the NHM IPM bulletin (2012), quoted in the module docstring, with our
interpretation documented. They output levels (High / Moderate / Low / Unknown) computed from hourly data. Missing
inputs give "Unknown".

**Back-test.** Krishnagiri, 2015–2024, 3,653 complete days.
- Open-Meteo's archive could not be reached from this environment: connection reset on every attempt, recorded
  in `source_attempts`. NASA POWER (MERRA-2, hourly, local solar time) was used instead.
- Every downloaded chunk is cached with its SHA-256.

**Results:**
- **Documented conditions at the "High" level:** 0.0 days/year for all three diseases.
- **Anthracnose.** Hourly RH ≥ 95% occurs 1,364 h/year, but never at 25–30 °C. RH ≥ 90% at 25–30 °C occurs
  2.6 h/year. The hour-by-hour reading of the bulletin's condition is therefore never met.
- **Powdery mildew.** Tmin of 10–13 °C occurs on only 4.4 days/year. Even ignoring RH, the condition holds on
  3.1 days/year.
- **"Moderate or High" days per year:**
  - bacterial canker: 165.9. This is a loose level, since one of the two conditions suffices.
  - powdery mildew: 47.3.
- **The previous heuristic rules** raised a "High" alert on 216.2 days/year. Their agreement with the documented
  conditions is κ = 0.0, with precision 0.0, because the documented conditions never occur.

**Can claim:**
- how often Krishnagiri reanalysis weather met the bulletin's favourable conditions, by month;
- that the previous rules disagreed with those conditions.

**Cannot claim:**
- any outbreak-prediction accuracy, lead time or false-alarm rate (no disease records exist);
- that the powdery-mildew thresholds, stated for the UP plains, hold in Tamil Nadu;
- orchard microclimate (coarse reanalysis grid);
- crop-stage relevance;
- that anthracnose-favourable weather never occurs at Krishnagiri (see above: co-occurrence in gridded data;
  canopy leaf wetness is not measured).

## 4. Tests
- `ml/severity/tests/test_ohi.py`: exclusion, renormalisation, Unknown status, unverified and off-host pests,
  inconsistency flag, worked example.
- `test_ohi_sensitivity.py` and `test_expert_agreement.py`.
- `ml/forecasting/tests/`: forecaster rules and back-test metrics.
- `ml/tests/test_full_pipeline.py`: untrained models give no OHI; offline weather is excluded; low-confidence
  handling; unverified and off-host pests.
- Backend `test_api.py`: null OHI, report rendering.
