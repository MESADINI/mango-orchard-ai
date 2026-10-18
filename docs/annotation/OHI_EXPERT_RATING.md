# Rating orchard health scenarios (OHI validation)

The Orchard Health Index (OHI) is a rule-based score. It has not yet been compared with expert judgement. This
sheet collects that comparison. No ratings exist yet.

## Who should rate
Agronomists, KVK or horticulture-department staff, or experienced mango growers. Each person rates on their own
copy of the sheet. Please do not discuss ratings before you have finished.

## The sheet
Run `python ml/severity/expert_agreement.py template` to create
`ml/datasets/annotation/ohi_expert_ratings_template.csv`. Each row is one scenario:

| Column | Meaning |
|---|---|
| `image` | the leaf photo (open it from the repository) |
| `predicted`, `confidence` | the disease classifier's output and its top probability |
| `severity_pct` | lesion area as a % of the leaf, from the colour segmenter (may include shadows or dirt) |
| `n_pests` | number of pest insects in the paired pest image |
| `rh`, `temp_c`, `precip_mm` | one hour of weather: relative humidity (%), temperature (°C), rain (mm) |

The OHI itself is hidden so that it cannot influence you.

## What to fill in
- `rater_id`: your initials or a code. Use the same one on every row.
- `rating_0_100`: how healthy would you judge a tree showing this, from 0 to 100?
  - 100 means fully healthy, with no action needed.
  - About 85 means minor issues, so monitor.
  - About 70 means treatment is advisable.
  - About 50 means a serious problem.
  - Below 50 means a critical outbreak.
- `notes`: optional. Use it if, for example, the photo is unusable or the prediction looks wrong.

Leave a row blank if you cannot judge it. At least 20 rated rows are needed.

## Afterwards
Return the file. `python ml/severity/expert_agreement.py evaluate rater1.csv rater2.csv ...` then computes:
- Spearman ρ between the OHI and the mean rating, with a bootstrap 95% CI;
- with two or more raters, the inter-rater agreement.

Results go to `results/ohi/expert_agreement.json`.
