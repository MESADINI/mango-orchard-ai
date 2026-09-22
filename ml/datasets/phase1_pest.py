#!/usr/bin/env python3
"""
AI Mango Orchard Health & Pest Intelligence — Phase 1 Pest Dataset Pipeline

This script downloads and standardizes confirmed pest datasets from Roboflow into YOLO format:
1. Fruit Fly (4,797 images): Roboflow 'school-vakkx/mango-fruit-fly' v4
2. Mealy Bug (881 images): Roboflow 'grape-n5yol/mealybug-b8wfq' v8
3. Scale Insect (809 images): Roboflow 'ralph-matthew-r-aquino/hemispherical-scale-insect' v1

Enforces Guardrails:
- Mango Hopper and Stem Borer have <50 images and are WEAK.
- They are excluded from YOLOv8 training until >=100 labeled bounding-box images exist.
- Surfaces warnings and points to ml/datasets/pest/MANUAL_ANNOTATION_GUIDE.md
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PEST_DIR = BASE_DIR / "processed" / "pest"

CONFIRMED_DATASETS = [
    {
        "name": "Fruit Fly",
        "workspace": "school-vakkx",
        "project": "mango-fruit-fly",
        "version": 4,
        "images": 4797,
        "status": "CONFIRMED"
    },
    {
        "name": "Mealy Bug",
        "workspace": "grape-n5yol",
        "project": "mealybug-b8wfq",
        "version": 8,
        "images": 881,
        "status": "CONFIRMED"
    },
    {
        "name": "Scale Insect",
        "workspace": "ralph-matthew-r-aquino",
        "project": "hemispherical-scale-insect",
        "version": 1,
        "images": 809,
        "status": "CONFIRMED"
    }
]

WEAK_DATASETS = [
    {
        "name": "Mango Hopper",
        "images": "< 50",
        "source": "None identified",
        "status": "WEAK — Needs manual annotation (threshold: >=100)"
    },
    {
        "name": "Stem Borer",
        "images": "< 50",
        "source": "None / IP102 transfer",
        "status": "WEAK — Needs manual annotation (threshold: >=100)"
    }
]

def check_roboflow_key() -> str:
    """Verifies that ROBOFLOW_API_KEY environment variable is set."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    return api_key or ""

def download_roboflow_dataset(item: Dict[str, Any], api_key: str, dest_dir: Path) -> bool:
    """Downloads a dataset version in YOLOv8 format using roboflow python SDK."""
    print(f"\n📦 Downloading '{item['name']}' from Roboflow ({item['workspace']}/{item['project']} v{item['version']})...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(item["workspace"]).project(item["project"])
        version = project.version(item["version"])
        dataset = version.download("yolov8", location=str(dest_dir / item["name"].lower().replace(" ", "_")))
        print(f"✅ Successfully downloaded {item['name']} dataset.")
        return True
    except Exception as e:
        print(f"❌ Failed downloading {item['name']}: {e}")
        return False

def print_pest_coverage_summary():
    """Prints the pest dataset coverage summary table."""
    print("=" * 80)
    print("🦗 Pest Dataset Coverage & Readiness Status")
    print("=" * 80)
    print(f"{'Pest Class':<16} | {'Coverage / Source':<42} | {'Readiness Status'}")
    print("-" * 80)

    for item in CONFIRMED_DATASETS:
        src = f"{item['workspace']}/{item['project']} v{item['version']} (~{item['images']})"
        print(f"{item['name']:<16} | {src:<42} | ✅ {item['status']}")

    for item in WEAK_DATASETS:
        src = f"{item['source']} ({item['images']} images)"
        print(f"{item['name']:<16} | {src:<42} | ⚠️  {item['status']}")
    print("-" * 80)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Mango Orchard Health & Pest Intelligence — Phase 1 Pest Ingestion")
    parser.add_argument("--sample", action="store_true", help="Generate sample YOLO pest dataset for testing")
    args = parser.parse_args()

    print("AI Mango Orchard Health & Pest Intelligence — Phase 1 Pest Ingestion")
    print_pest_coverage_summary()

    api_key = check_roboflow_key()
    if not api_key:
        print("\n⚠️  ROBOFLOW_API_KEY environment variable is not set.")
        print("To download confirmed pest datasets automatically:")
        print("  1. Create a free account at https://roboflow.com")
        print("  2. Go to Settings -> API Keys -> copy your private key")
        print("  3. Run: export ROBOFLOW_API_KEY='your_api_key'")
        print("  4. Re-run: python ml/datasets/phase1_pest.py")
        print("\n⚠️  Policy Reminder:")
        print("  Mango Hopper & Stem Borer must NOT be trained until >=100 labeled bounding-box images exist.")
        print("  Refer to ml/datasets/pest/MANUAL_ANNOTATION_GUIDE.md for the annotation procedure.\n")

        # Check if sample pest data already exists or if sample mode requested
        fruit_fly_dir = PEST_DIR / "fruit_fly" / "train" / "images"
        if fruit_fly_dir.exists() and len(list(fruit_fly_dir.glob("*.jpg"))) > 0:
            print("Found existing pest datasets.")
        elif args.sample or not api_key:
            print("Generating structured YOLO sample pest dataset for immediate pipeline testing...")
            try:
                from ml.datasets.generate_sample_data import generate_pest_dataset
                generate_pest_dataset(images_per_class=15)
            except Exception as e:
                print(f"Pest sample generation notice: {e}")
        return

    PEST_DIR.mkdir(parents=True, exist_ok=True)
    for item in CONFIRMED_DATASETS:
        download_roboflow_dataset(item, api_key, PEST_DIR)

if __name__ == "__main__":
    main()
