#!/usr/bin/env python3
"""
AI Mango Orchard Health & Pest Intelligence — YOLOv8 Pest Dataset Preparation
Architecture: Ultralytics YOLOv8
Confirmed Classes: Fruit Fly (0), Mealy Bug (1), Scale Insect (2)
Inviolable Rule 3: Mango Hopper and Stem Borer must NOT be included in YOLOv8 training.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageDraw

CONFIRMED_CLASSES = ["Fruit Fly", "Mealy Bug", "Scale Insect"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CONFIRMED_CLASSES)}

UNRELIABLE_CLASSES = ["Mango Hopper", "Stem Borer"]

BASE_DIR = Path(__file__).resolve().parent
PEST_DIR = BASE_DIR / "processed" / "pest"
SAMPLE_PESTS_DIR = BASE_DIR / "sample_pests"

def generate_pest_image_with_annotations(pest_name: str, width: int = 640, height: int = 640):
    """
    Generates a realistic orchard foliage image with pest silhouettes and bounding boxes.
    """
    img = Image.new("RGB", (width, height), color=(85, 125, 65))
    draw = ImageDraw.Draw(img)

    # Orchard canopy leaves texture
    for _ in range(12):
        lx = random.randint(30, width - 120)
        ly = random.randint(30, height - 120)
        lw = random.randint(100, 220)
        lh = random.randint(60, 140)
        leaf_col = (random.randint(35, 70), random.randint(110, 160), random.randint(25, 55))
        draw.ellipse([lx, ly, lx + lw, ly + lh], fill=leaf_col, outline=(30, 90, 20))

    bboxes = []
    num_pests = random.randint(1, 3)
    for _ in range(num_pests):
        pw = random.randint(45, 90)
        ph = random.randint(45, 90)
        cx = random.randint(pw, width - pw)
        cy = random.randint(ph, height - ph)

        x1 = cx - pw // 2
        y1 = cy - ph // 2
        x2 = cx + pw // 2
        y2 = cy + ph // 2

        if pest_name == "Fruit Fly":
            draw.ellipse([x1, y1, x2, y2], fill=(210, 140, 30), outline=(130, 80, 10))
            draw.ellipse([x1 + 5, y1 + 5, x1 + pw // 2, y1 + ph // 2], fill=(240, 220, 150))
            draw.ellipse([x2 - pw // 2, y1 + 5, x2 - 5, y1 + ph // 2], fill=(240, 220, 150))
        elif pest_name == "Mealy Bug":
            draw.ellipse([x1, y1, x2, y2], fill=(245, 245, 245), outline=(210, 210, 210))
            for i in range(x1, x2, 8):
                draw.line([(i, y1), (i, y1 - 4)], fill=(230, 230, 230), width=2)
                draw.line([(i, y2), (i, y2 + 4)], fill=(230, 230, 230), width=2)
        elif pest_name == "Scale Insect":
            draw.ellipse([x1, y1, x2, y2], fill=(140, 85, 40), outline=(90, 50, 20))
            draw.ellipse([cx - pw // 4, cy - ph // 4, cx + pw // 4, cy + ph // 4], fill=(175, 115, 55))
        elif pest_name == "Mango Hopper":
            draw.polygon([(cx, y1), (x2, y2), (x1, y2)], fill=(160, 130, 90), outline=(100, 80, 50))
        elif pest_name == "Stem Borer":
            draw.rectangle([x1, y1, x2, y2], fill=(70, 50, 40), outline=(40, 30, 20))
            draw.line([(cx, y1), (x1 - 10, y1 - 20)], fill=(50, 40, 30), width=2)
            draw.line([(cx, y1), (x2 + 10, y1 - 20)], fill=(50, 40, 30), width=2)

        x_c_norm = cx / width
        y_c_norm = cy / height
        w_norm = pw / width
        h_norm = ph / height
        bboxes.append((x_c_norm, y_c_norm, w_norm, h_norm))

    return img, bboxes

def prepare_yolo_pest_dataset(samples_per_class: int = 25):
    print("=" * 70)
    print("🦗 Preparing Consolidated YOLOv8 Pest Dataset")
    print(f"Confirmed Training Classes: {CONFIRMED_CLASSES}")
    print(f"Gated Unreliable Classes (Excluded from Training): {UNRELIABLE_CLASSES}")
    print("=" * 70)

    images_train = PEST_DIR / "images" / "train"
    images_val = PEST_DIR / "images" / "val"
    labels_train = PEST_DIR / "labels" / "train"
    labels_val = PEST_DIR / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val, SAMPLE_PESTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    train_count = int(samples_per_class * 0.8)
    val_count = samples_per_class - train_count

    for cls_name in CONFIRMED_CLASSES:
        cls_idx = CLASS_TO_IDX[cls_name]
        slug = cls_name.lower().replace(" ", "_")

        for split, count, img_dir, lbl_dir in [
            ("train", train_count, images_train, labels_train),
            ("val", val_count, images_val, labels_val)
        ]:
            for i in range(count):
                img, bboxes = generate_pest_image_with_annotations(cls_name)
                img_name = f"{slug}_{split}_{i:03d}.jpg"
                lbl_name = f"{slug}_{split}_{i:03d}.txt"

                img.save(img_dir / img_name, "JPEG", quality=90)
                with open(lbl_dir / lbl_name, "w") as f:
                    for x_c, y_c, w, h in bboxes:
                        f.write(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    print("\n📸 Generating representative sample images for testing & validation...")
    for cls_name in CONFIRMED_CLASSES + UNRELIABLE_CLASSES:
        slug = cls_name.lower().replace(" ", "_")
        img, _ = generate_pest_image_with_annotations(cls_name)
        img.save(SAMPLE_PESTS_DIR / f"{slug}_sample.jpg", "JPEG", quality=95)

    clean_img = Image.new("RGB", (640, 640), color=(85, 125, 65))
    draw = ImageDraw.Draw(clean_img)
    for _ in range(16):
        lx = random.randint(20, 500)
        ly = random.randint(20, 500)
        lw = random.randint(100, 220)
        lh = random.randint(60, 140)
        draw.ellipse([lx, ly, lx + lw, ly + lh], fill=(random.randint(40, 75), random.randint(110, 160), 35))
    clean_img.save(SAMPLE_PESTS_DIR / "clean_canopy_sample.jpg", "JPEG", quality=95)

    data_yaml_path = PEST_DIR / "data.yaml"
    data_yaml_content = f"""path: {PEST_DIR.resolve()}
train: images/train
val: images/val

nc: {len(CONFIRMED_CLASSES)}
names: {CONFIRMED_CLASSES}
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)

    print(f"✅ data.yaml written to: {data_yaml_path}")
    print(f"✅ Pest dataset successfully created in: {PEST_DIR}")
    print(f"   - Training images: {len(list(images_train.glob('*.jpg')))}")
    print(f"   - Validation images: {len(list(images_val.glob('*.jpg')))}")
    print(f"   - Sample images in: {SAMPLE_PESTS_DIR}")

if __name__ == "__main__":
    prepare_yolo_pest_dataset(samples_per_class=25)
