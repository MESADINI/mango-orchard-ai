# Field images (to be collected)

This folder is for **your own photos from real orchards** (e.g. Krishnagiri / Dharmapuri). They are the strongest
test of whether the models generalise. Nothing here is used for training unless you decide so explicitly.

## How to add images

1. Put photos in `ml/datasets/field/images/<Class>/`, where `<Class>` is one of
   `Anthracnose`, `Bacterial Canker`, `Powdery Mildew`, `Sooty Mold`, `Healthy`,
   or `Other` for any disease the model does not know (used to test the "unsure, retake photo" rejection).
2. Record one row per image in `ml/datasets/field/labels.csv` (template: `labels_template.csv`):
   `filename, label, labelled_by, label_confidence (high/medium/low), date, location, device, notes`.
   The label should come from an agronomist / plant pathologist where possible (`labelled_by`).
3. One leaf per photo, filling most of the frame; note in `notes` if the background is cluttered.
4. Do not edit, crop or filter the photos.

`images/` is git-ignored (photos can be large and may contain location metadata); `labels.csv` is committed.

## Suggested minimum

At least 50 photos per known class plus 50 `Other`, spread over several orchards, days and phones.
Pest-trap / canopy photos for the pest detector go in `ml/datasets/field/pest_images/` with YOLO labels.
