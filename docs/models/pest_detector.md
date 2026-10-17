# Model card: pest detector (proxy data, host-unverified)

All numbers are copied from `results/pest/`. Registry entry: `ml/model_registry.json` (`pest_detector`).

| | |
|---|---|
| Architecture | YOLO11n (Ultralytics 8.3.40), 2.6 M parameters, COCO-pretrained |
| Classes | Fruit Fly, Mealy Bug, Scale Insect, all `host_verified: false` |
| Weights | `ml/pest_detection/weights/pest_detector.pt`, SHA-256 `dfd2d0e2…4315` |
| Selection | highest val mAP50-95 of YOLOv8n and YOLO11n (YOLOv8s not completed) |
| Test mAP50 / mAP50-95 | 0.744 / 0.520 (n = 379 images) |
| Test AP50-95 per class | Fruit Fly 0.542, Mealy Bug 0.121, Scale Insect 0.896 |

## Training data
Three public Roboflow sets (CC BY 4.0), one per class; card: `docs/datasets/pest.md`. None of them shows mango:

| Class | Source images |
|---|---|
| Fruit Fly | grayscale trap photos; the species could not be confirmed as *Bactrocera* |
| Mealy Bug | grape bunches and stock photos; not *Drosicha mangiferae* |
| Scale Insect | *Saissetia coffeae* on coffee leaves |

## Intended use and limits
- **Use:** research and demonstration only. In the app:
  - every detection carries an "unverified: not trained on mango" warning;
  - the pest component is excluded from the OHI;
  - no treatment is recommended from these detections.
- **Do not interpret the mAP as orchard accuracy.** Each class is confounded with its source domain.
- **Mealy Bug** detection is poor even in-domain: test AP50 0.26.
- **Not detected:** Mango Hopper and Stem Borer are gated, with < 100 annotations.
- **Needed to change this status:** annotated images of these pests on mango trees, from traps and canopy photos
  in Indian orchards. With those, a class's `host_verified` flag can be set by the review in
  `ml/datasets/pest_source_review.json`.
