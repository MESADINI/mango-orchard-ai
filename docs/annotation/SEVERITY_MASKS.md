# Annotating lesion masks for severity validation

The lesion-area estimate (`ml/severity/severity.py`) is a hand-tuned colour rule. Its accuracy is unknown until it is
compared with masks drawn by a person. `ml/severity/validate.py evaluate` reports nothing until **at least 50**
masks exist.

## What to annotate
`ml/datasets/annotation/severity_sample.csv` lists 60 test-split images: 12 per class, one image per leaf. Annotate
all 60 if you can; 50 is the minimum. Do not swap in other images, because the sample is fixed so that the result can
be reproduced. The images are in `ml/datasets/processed/disease/` (run `python tasks.py data` first).

## Tool: LabelMe (free)
1. `pip install labelme`, then `labelme ml/datasets/processed/disease/test --output ml/datasets/annotation/labelme`
2. For each image in the sample list, draw polygons with exactly these labels:
   - `leaf`: the outline of the whole leaf, including diseased parts and the petiole if visible.
   - `lesion`: each diseased area: brown or black necrosis, cankers, the black sooty coating, and the white powdery
     coating. Draw one polygon per lesion. Small spots under ~3 px can be skipped.
3. Save. LabelMe writes one `.json` per image.
4. Convert: `python ml/severity/validate.py convert ml/datasets/annotation/labelme`
   This writes `ml/datasets/annotation/severity_masks/<stem>.png` (0 = background, 128 = leaf, 255 = lesion).

Masks drawn in another tool (CVAT, GIMP) are fine if they use the same three grey levels and the image's file stem.

## Rules
- **Healthy leaves:** draw only `leaf`, unless you see a real lesion.
- **Shadows, dirt and glare:** these are not lesions.
- **Yellow halo around a lesion:** not lesion (only the necrotic or coated tissue counts).
- **Unsure:** if you cannot tell whether an area is diseased, leave it out and note the file name in
  `ml/datasets/annotation/notes.txt`.
- **Second annotator (optional, recommended):** a second person annotates 15 of the same images, so that the
  agreement between people can be compared with the agreement between person and algorithm.

## Evaluate
`python ml/severity/validate.py evaluate` writes `results/severity/validation.json` and `results/severity/per_image.csv`.
These contain lesion IoU/Dice, leaf IoU, severity MAE (percentage points) and Spearman ρ, per class and overall, with
bootstrap 95% CIs.
