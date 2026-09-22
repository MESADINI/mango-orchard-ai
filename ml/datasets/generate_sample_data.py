#!/usr/bin/env python3
"""
AI Mango Orchard Health & Pest Intelligence — Sample & Synthetic Dataset Generator

Generates verified sample datasets with exact dimensions and directory structure:
1. Disease classes: Anthracnose, Bacterial Canker, Powdery Mildew, Sooty Mold, Leaf Spot, Healthy
   Splits: train (70%), val (15%), test (15%)
2. Confirmed Pest classes in YOLOv8 structure: Fruit Fly, Mealy Bug, Scale Insect
3. Sample leaves for Grad-CAM explainability and severity segmentation testing.
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DISEASE_DIR = BASE_DIR / "processed" / "disease"
PROCESSED_PEST_DIR = BASE_DIR / "processed" / "pest"
SAMPLE_LEAVES_DIR = BASE_DIR / "sample_leaves"

DISEASE_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Powdery Mildew",
    "Sooty Mold",
    "Leaf Spot",
    "Healthy"
]

PEST_CLASSES = [
    "Fruit Fly",
    "Mealy Bug",
    "Scale Insect"
]

def create_base_leaf_canvas(width: int = 300, height: int = 300) -> Image.Image:
    """Creates a realistic mango leaf silhouette on an orchard background."""
    img = Image.new("RGB", (width, height), color=(235, 230, 220))
    draw = ImageDraw.Draw(img)

    # Leaf blade (elliptic polygon)
    leaf_color = (34, 139, 34)  # Forest green
    points = [
        (width // 2, 20),
        (width - 40, height // 3),
        (width - 50, 2 * height // 3),
        (width // 2, height - 30),
        (50, 2 * height // 3),
        (40, height // 3)
    ]
    draw.polygon(points, fill=leaf_color, outline=(24, 100, 24))

    # Midrib vein
    draw.line([(width // 2, 20), (width // 2, height - 30)], fill=(46, 160, 46), width=3)
    # Secondary veins
    for y in range(60, height - 50, 30):
        draw.line([(width // 2, y), (width - 70, y - 15)], fill=(40, 150, 40), width=1)
        draw.line([(width // 2, y), (70, y - 15)], fill=(40, 150, 40), width=1)

    return img

def render_disease_symptoms(img: Image.Image, disease_name: str) -> Image.Image:
    """Draws representative pathogen symptoms onto the leaf."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if disease_name == "Healthy":
        # Slight natural color gradation
        return img

    elif disease_name == "Anthracnose":
        # Dark brown / black irregular necrotic spots with jagged edges
        for _ in range(random.randint(6, 14)):
            cx = random.randint(80, w - 80)
            cy = random.randint(60, h - 80)
            r = random.randint(8, 22)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(45, 25, 15), outline=(20, 10, 5))
            # Necrotic center
            draw.ellipse([cx - r//2, cy - r//2, cx + r//2, cy + r//2], fill=(20, 10, 5))

    elif disease_name == "Bacterial Canker":
        # Angular water-soaked lesions with yellow chlorotic halo
        for _ in range(random.randint(5, 10)):
            cx = random.randint(80, w - 80)
            cy = random.randint(60, h - 80)
            r = random.randint(10, 20)
            # Yellow chlorotic halo
            draw.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], fill=(210, 195, 30))
            # Dark brown canker center
            draw.rectangle([cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2], fill=(60, 30, 10))

    elif disease_name == "Powdery Mildew":
        # Superficial white/ash-gray powdery patches
        for _ in range(random.randint(7, 15)):
            cx = random.randint(70, w - 70)
            cy = random.randint(50, h - 70)
            r = random.randint(12, 28)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(225, 230, 230))

    elif disease_name == "Sooty Mold":
        # Diffuse velvety black coating along leaf blade
        for _ in range(random.randint(15, 25)):
            cx = random.randint(60, w - 60)
            cy = random.randint(50, h - 60)
            r = random.randint(15, 35)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(25, 25, 25))

    elif disease_name == "Leaf Spot":
        # Well-defined circular brown spots with dark concentric rings
        for _ in range(random.randint(8, 16)):
            cx = random.randint(80, w - 80)
            cy = random.randint(60, h - 80)
            r = random.randint(6, 14)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120, 60, 20), outline=(50, 25, 10))
            draw.ellipse([cx - r//3, cy - r//3, cx + r//3, cy + r//3], fill=(80, 40, 15))

    return img.filter(ImageFilter.SMOOTH_MORE)

def generate_disease_dataset(images_per_class: int = 20):
    """Generates stratified disease image dataset."""
    print(f"🌿 Generating disease dataset ({images_per_class} images/class)...")
    splits = {
        "train": int(images_per_class * 0.70),
        "val": int(images_per_class * 0.15),
        "test": images_per_class - int(images_per_class * 0.70) - int(images_per_class * 0.15)
    }

    for split_name, count in splits.items():
        for cls_name in DISEASE_CLASSES:
            cls_dir = PROCESSED_DISEASE_DIR / split_name / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                leaf = create_base_leaf_canvas(300, 300)
                leaf = render_disease_symptoms(leaf, cls_name)
                filename = f"{cls_name.lower().replace(' ', '_')}_{split_name}_{i:03d}.jpg"
                leaf.save(cls_dir / filename, "JPEG", quality=90)

    # Also save one representative image per class into sample_leaves
    SAMPLE_LEAVES_DIR.mkdir(parents=True, exist_ok=True)
    for cls_name in DISEASE_CLASSES:
        leaf = create_base_leaf_canvas(300, 300)
        leaf = render_disease_symptoms(leaf, cls_name)
        leaf.save(SAMPLE_LEAVES_DIR / f"{cls_name.lower().replace(' ', '_')}_sample.jpg", "JPEG", quality=95)

    print(f"✅ Generated disease datasets across {len(DISEASE_CLASSES)} classes in {PROCESSED_DISEASE_DIR}")

def generate_pest_dataset(images_per_class: int = 15):
    """Generates YOLOv8 formatted sample datasets for confirmed pests."""
    print(f"🦗 Generating YOLO pest dataset ({images_per_class} images/class)...")

    for idx, pest_name in enumerate(PEST_CLASSES):
        pest_folder = pest_name.lower().replace(" ", "_")
        for split in ["train", "val"]:
            img_dir = PROCESSED_PEST_DIR / pest_folder / split / "images"
            lbl_dir = PROCESSED_PEST_DIR / pest_folder / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            count = int(images_per_class * 0.8) if split == "train" else max(3, int(images_per_class * 0.2))
            for i in range(count):
                w, h = 640, 640
                img = Image.new("RGB", (w, h), color=(100, 140, 80))
                draw = ImageDraw.Draw(img)

                # Draw background foliage
                draw.ellipse([50, 50, 590, 590], fill=(60, 120, 50))

                # Draw pest silhouette
                cx, cy = random.randint(150, 490), random.randint(150, 490)
                bw, bh = random.randint(40, 80), random.randint(40, 80)
                color = (200, 140, 30) if pest_name == "Fruit Fly" else (240, 240, 240) if pest_name == "Mealy Bug" else (160, 90, 40)
                draw.ellipse([cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2], fill=color)

                img_path = img_dir / f"{pest_folder}_{split}_{i:03d}.jpg"
                img.save(img_path, "JPEG")

                # YOLO annotation: <class_id> <x_center> <y_center> <width> <height>
                lbl_path = lbl_dir / f"{pest_folder}_{split}_{i:03d}.txt"
                with open(lbl_path, "w") as f:
                    f.write(f"{idx} {cx/w:.5f} {cy/h:.5f} {bw/w:.5f} {bh/h:.5f}\n")

    print(f"✅ Generated YOLO pest datasets in {PROCESSED_PEST_DIR}")

def main():
    print("====================================================================")
    print("🥭 Generating Sample Data for AI Mango Orchard Health Intelligence")
    print("====================================================================")
    generate_disease_dataset(images_per_class=24)
    generate_pest_dataset(images_per_class=15)
    print("\n✅ Sample data generation complete.")

if __name__ == "__main__":
    main()
