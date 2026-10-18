# Orchard Health Index (OHI) Methodology

This document describes exactly what `ml/severity/ohi.py` computes. If the two disagree, the code is the
reference and this file is wrong. The test `ml/severity/tests/test_ohi.py` checks the worked numbers below.

## 1. What the OHI is and is not
- The OHI is a **deterministic, rule-based** score from 0 to 100, where 100 means no penalty.
- It combines up to four component penalties with fixed weights.
- Every constant in it is **expert-assigned, not fitted to data**: the weights, the virulence factors α, the pest
  impact factors β, the weather curve and the tier bounds.
- It has **not been validated against expert judgement or yield**. A rating protocol is ready
  (`docs/annotation/OHI_EXPERT_RATING.md`), but no ratings exist yet.
- How strongly the constants drive the score is measured by the sensitivity analysis in §6, not assumed.

## 2. Formula
With the set of available components A ⊆ {disease, pest, weather, nutrient}:

$$\text{OHI} = \operatorname{clip}_{[0,100]}\Big(100 - \sum_{i \in A} \tilde w_i \, X_i\Big), \qquad
\tilde w_i = \frac{w_i}{\sum_{j \in A} w_j}$$

Here $X_i \in [0, 100]$ is the penalty of component i.

### Nominal weights

| Component | Penalty | $w_i$ |
|---|---|---|
| Disease severity | $S_d$ | 0.40 |
| Pest pressure | $P_p$ | 0.35 |
| Weather vulnerability | $W_v$ | 0.15 |
| Nutrient deficiency | $N_d$ | 0.10 |

### Missing components are excluded, never scored as healthy
A component is missing when:
- the scan type skipped it;
- no trained model is installed;
- the disease prediction is below the calibrated low-confidence threshold (the app then asks the user to retake
  the photo);
- the pest detector has no class trained on mango images (§3B);
- live weather is unavailable;
- no nutrient sensor exists. This is always the case today.

The remaining weights are rescaled proportionally. With today's inputs (nutrient always missing), the effective
weights are:
- disease 0.40/0.90 = 0.444;
- pest 0.35/0.90 = 0.389;
- weather 0.15/0.90 = 0.167.

The response lists the missing components (`missing_components`, `is_partial`) and the weights actually used
(`weights_used`).

If neither image component (disease or pest) is available, **no OHI is reported**: `ohi_score` is null and the
status is "Unknown — insufficient data". A weather-only number would look like a health assessment without being
one.

## 3. Component penalties

### A. Disease, $S_d$
The disease class c is one of the five model classes. Leaf Spot is not a class, because no licensed training data
exists for it.

$$S_d = \begin{cases} 0 & c = \text{Healthy} \\
\min\big(100,\; (40\,\alpha_c + 0.6\, s) \cdot p_c\big) & \text{otherwise} \end{cases}$$

- $s \in [0, 100]$ is the lesion area as a % of the leaf, from the HSV segmenter (`ml/severity/severity.py`).
- $p_c \in [0, 1]$ is the classifier's top probability under the deployed temperature.
- The virulence factors are:
  - Anthracnose α = 1.00;
  - Bacterial Canker α = 0.95;
  - Powdery Mildew α = 0.85;
  - Sooty Mold α = 0.50.
- An unknown class name raises `ValueError`. It never falls back to a default.

**Healthy with lesions.** If c = Healthy but s ≥ 5%, the scan is flagged as `healthy_lesion_inconsistency`:
- no disease penalty is applied;
- the caveat asks for a retake or an expert check.

The 5% value is a rule of thumb, because the segmenter also marks shadows and dirt.

### B. Pest, $P_p$

$$P_p = \min\Big(100,\; \sum_{\text{detections } i} \beta_{k_i} \, q_i\Big)$$

- $q_i$ is the detection confidence.
- The impact factors are:
  - Fruit Fly β = 18;
  - Mealy Bug β = 12;
  - Scale Insect β = 10;
  - Mango Hopper β = 15;
  - Stem Borer β = 35.
- There is no canopy-density or other attenuation term.
- An unknown class raises `ValueError`.

Some detections are shown to the user but **excluded from the sum**:
- **Mango Hopper and Stem Borer** (`is_unreliable`): they have fewer than 100 annotated training samples.
- **Classes trained on non-mango images** (`host_verified: false`). The deployed detector's training images come
  from grayscale trap or petri-dish photos (fruit flies), grape and stock photos (mealybugs) and coffee leaves (scale
  insects); see `ml/datasets/pest_source_review.json`. All three classes are therefore host-unverified.

Because no class of the deployed detector is host-verified, the pipeline also treats the **pest component as
missing**. Finding no pest with this model is not evidence of a pest-free mango tree. `excluded_unverified_pest_detections`
counts the excluded boxes.

### C. Weather, $W_v$
The inputs are the current hour's values from Open-Meteo: relative humidity RH (%), temperature T (°C) and
precipitation R (mm).

$$f_{RH} = \operatorname{clip}_{[0,100]}\big(2\,(RH - 50)\big), \quad
f_T = \begin{cases} 100 - 10\,|T - 27| & 22 \le T \le 32 \\ 20 & \text{otherwise} \end{cases}, \quad
f_R = \operatorname{clip}_{[0,100]}(15\, R)$$

$$W_v = \operatorname{clip}_{[0,100]}\big(0.45\, f_{RH} + 0.35\, f_T + 0.20\, f_R\big)$$

- There is no 2 mm rain threshold. Rain risk rises linearly and saturates at 6.67 mm.
- The curve is a heuristic, not a disease model.
- The 7-day disease forecast uses separate, cited rules (`ml/forecasting/forecaster.py`); see §7.

### D. Nutrient, $N_d$
- No soil or leaf sensor is integrated, so $N_d$ is always missing.
- It is excluded and the other weights are renormalised. It is not set to 0.
- The response sets `nutrient_input_was_defaulted: true`.

## 4. Tiers
The tier bounds are expert-assigned.

| OHI | Tier | Colour |
|---|---|---|
| ≥ 85 | Optimal Health | `#10B981` |
| 70 – < 85 | Mild Stress / Warning | `#F59E0B` |
| 50 – < 70 | High Infestation Risk | `#F97316` |
| < 50 | Severe Outbreak / Critical | `#EF4444` |
| no OHI | Unknown — insufficient data | `#6B7280` |

## 5. Worked example
**Inputs:**
- disease: Anthracnose, p = 0.90, s = 20%;
- pest: one verified Fruit Fly box at q = 0.80;
- weather: RH 85%, T 26 °C, R 0 mm;
- nutrient: missing.

**Penalties:**
- $S_d = (40 + 12) \cdot 0.90 = 46.8$.
- $P_p = 18 \cdot 0.80 = 14.4$.
- $W_v = 0.45 \cdot 70 + 0.35 \cdot 90 + 0 = 63.0$.

**Score:** OHI = 100 − (0.4444·46.8 + 0.3889·14.4 + 0.1667·63.0) = 100 − 36.90 = **63.10**, which is in the
"High Infestation Risk" tier.

## 6. Sensitivity analysis
`ml/severity/ohi_sensitivity.py` writes `results/ohi/sensitivity.{json,md}`. It covers:
- one-at-a-time ±10/20% changes to each weight, and ±20% to each α and β;
- Monte Carlo Dirichlet weights;
- component ablations.

These run on 239 MangoLeafBD test scenarios, scored with the deployed classifier's predictions.

**Results** (copied from `results/ohi/sensitivity.md`; seed 42, 2,000 Monte Carlo draws):
- **Ranking is robust.** Scaling any one weight by 0.8–1.2 keeps Spearman ρ ≥ 0.993 against the nominal scores.
  A ±20% change to any α or β keeps ρ ≥ 0.994.
- **Tier labels are not robust to the weights.**
  - One-at-a-time changes move up to 12.1% of scenarios into another tier.
  - Under Dirichlet weights centred on the nominal ones (c = 100), a median of 7.5% of tiers change; at c = 20
    the median is 17.2%.
  - With no prior on the weights (uniform Dirichlet), a median of 42.3% of tiers change.
  - The tier boundaries therefore rest on the expert-chosen weights, which is why expert validation is needed.
- **Ablation.** Removing disease or pest changes tiers for 48.1% and 56.4% of scenarios. Removing weather changes
  18.8% (Spearman ρ 0.987). The weather term mainly shifts scores rather than reordering them.
- **Scenario caveat.** The pest and weather inputs are paired synthetically with the leaf images, and the pest
  term is scored as if it were verified. In the deployed app the pest component is currently excluded (§3B).

## 7. Relation to the weather forecast
The OHI weather term (§3C) scores **one hour of current weather**. The 7-day outlook in the app is a separate rule
set:
- thresholds from the NHM mango IPM bulletin (2012);
- outputs are levels (High / Moderate / Low / Unknown);
- it does not enter the OHI.

`ml/forecasting/backtest.py` reports what that rule set can and cannot claim.
