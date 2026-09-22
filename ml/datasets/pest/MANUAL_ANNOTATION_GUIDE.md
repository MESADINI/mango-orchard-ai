# Manual Annotation Guide: Mango Hopper & Stem Borer

## 1. Why Manual Annotation is Required
In our Phase 1 dataset discovery audit, two high-impact pest species exhibited critically insufficient public training samples:
- **Mango Hopper** (*Amritodus atkinsoni* / *Idioscopus clypealis*): $<50$ confirmed images.
- **Stem Borer** (*Batocera rufomaculata*): $<50$ confirmed images.

> [!WARNING]
> **Strict Training Guardrail**:
> Training an object detection model (YOLOv8) on fewer than 100 high-quality bounding-box instances yields severe false-positive rates, poor generalizeability, and hallucinations.
> Therefore, both classes are designated as `UNRELIABLE_CLASSES` in `ml/pest_detection/inference.py` and must **NOT** be included in model weights until $\ge 100$ verified instances are annotated per class.

---

## 2. Annotation Specifications

### Tool Selection
Use **Roboflow Annotate** or **CVAT (Computer Vision Annotation Tool)**:
- Output Format: **YOLOv8 PyTorch TXT** format (`<class-index> <x_center> <y_center> <width> <height>` normalized $[0, 1]$).

### Class Index Mapping
```
0: Fruit Fly
1: Mealy Bug
2: Scale Insect
3: Mango Hopper      (Requires >= 100 annotations to unlock)
4: Stem Borer        (Requires >= 100 annotations to unlock)
```

---

## 3. Labeling Criteria

### A. Mango Hopper (*Amritodus atkinsoni*)
- **Target Morphology**: Wedge-shaped leafhopper, 4-5 mm long, greyish-brown with dark markings on head/pronotum.
- **Key Habitat on Tree**: Clustered on flower panicles, tender rachis, and underside of young flush leaves.
- **Bounding Box Rules**:
  - Draw tight bounding boxes enclosing the entire adult insect including wings.
  - If nymphs are present in dense aggregations, box individual nymphs where distinct; do not create a single giant box over the entire inflorescence unless labeling a cluster sub-class.
  - Exclude honeydew droplet glistening from the bounding box.

### B. Stem Borer (*Batocera rufomaculata*)
- **Target Morphology**: Large longhorn beetle (50-55 mm), dark brownish-grey with yellowish-orange spots on elytra and long antennae.
- **Larval Damage vs Adult**:
  - **Adult beetle**: Draw bounding box around entire body including antennae roots.
  - **Frass / Bore Holes**: If annotating bore damage on tree bark, tag as `Stem Borer Borehole` rather than insect entity to avoid confusing the detector.

---

## 4. Verification & Ingestion Checklist
1. Gather $\ge 100$ raw images across varying daylight angles, canopy depths, and flash conditions.
2. Label in Roboflow / CVAT following the tight bounding box standards above.
3. Export dataset in YOLOv8 format.
4. Place files in `ml/datasets/processed/pest/mango_hopper/` and `ml/datasets/processed/pest/stem_borer/`.
5. Run `python ml/pest_detection/train.py --include-all` once verification threshold is fulfilled.
