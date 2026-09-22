#!/usr/bin/env python3
"""
AI Mango Orchard Health & Pest Intelligence — Phase 1 Disease Dataset Pipeline

This script downloads and preprocesses the disease detection dataset:
1. Primary Source: MangoLeafBD from Kaggle
2. Class mapping & filtering:
   - Anthracnose -> Anthracnose (Keep)
   - Bacterial Canker -> Bacterial Canker (Keep)
   - Powdery Mildew -> Powdery Mildew (Keep)
   - Sooty Mould -> Sooty Mold (Rename)
   - Healthy -> Healthy (Keep)
   - Cutting Weevil, Die Back, Gall Midge -> Exclude
3. Leaf Spot supplemental dataset acquisition hook
4. Verification of image integrity & stratified split generation (70/15/15)
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Base paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = BASE_DIR / "raw" / "disease"
PROCESSED_DIR = BASE_DIR / "processed" / "disease"

# Class Mapping Schema
CLASS_MAPPING = {
    "anthracnose": "Anthracnose",
    "bacterial canker": "Bacterial Canker",
    "powdery mildew": "Powdery Mildew",
    "sooty mould": "Sooty Mold",
    "healthy": "Healthy"
}

EXCLUDED_CLASSES = [
    "cutting weevil",
    "die back",
    "gall midge"
]

TARGET_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Powdery Mildew",
    "Sooty Mold",
    "Leaf Spot",
    "Healthy"
]

def check_kaggle_credentials() -> bool:
    """Verifies that kaggle.json exists or environment variables are present."""
    kaggle_home = Path.home() / ".kaggle" / "kaggle.json"
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    if kaggle_home.exists() or has_env:
        return True
    return False

def download_mangoleafbd(target_dir: Path) -> bool:
    """Downloads MangoLeafBD dataset using Kaggle API."""
    print("=" * 70)
    print("🥭 Step 1: Downloading MangoLeafBD Dataset from Kaggle")
    print("=" * 70)
    
    if not check_kaggle_credentials():
        print("\n⚠️  Kaggle API credentials not found!")
        print("To download the dataset automatically:")
        print("  1. Go to https://www.kaggle.com/settings -> API -> 'Create New Token'")
        print(f"  2. Save 'kaggle.json' to: {Path.home()}/.kaggle/kaggle.json")
        print("  3. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nAlternatively, set environment variables:")
        print("  export KAGGLE_USERNAME='your_username'")
        print("  export KAGGLE_KEY='your_api_key'")
        return False
        
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        target_dir.mkdir(parents=True, exist_ok=True)
        dataset_slug = "aryashah2k/mango-leaf-disease-dataset"
        print(f"Downloading dataset '{dataset_slug}' into {target_dir}...")
        api.dataset_download_files(dataset_slug, path=str(target_dir), unzip=True)
        print("✅ Download and extraction complete.")
        return True
    except Exception as e:
        print(f"❌ Failed to download via Kaggle API: {e}")
        return False

def process_and_filter_classes(raw_dir: Path, processed_dir: Path) -> Dict[str, int]:
    """Applies schema class mapping, renames, and excludes non-target classes."""
    print("\n" + "=" * 70)
    print("🌿 Step 2: Applying Class Mapping & Exclusion Rules")
    print("=" * 70)

    # Prepare directories for target classes
    for split in ["train", "val", "test"]:
        for cls_name in TARGET_CLASSES:
            (processed_dir / split / cls_name).mkdir(parents=True, exist_ok=True)

    class_counts: Dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    collected_files: Dict[str, List[Path]] = {c: [] for c in TARGET_CLASSES}

    # Find raw image directories
    for root, dirs, files in os.walk(raw_dir):
        folder_name = Path(root).name.lower().strip()
        matched_target = None

        # Check exclusion
        if any(exc in folder_name for exc in EXCLUDED_CLASSES):
            print(f"  🚫 Excluding non-disease class: '{folder_name}' ({len(files)} items)")
            continue

        # Check mapping
        for src_key, target_val in CLASS_MAPPING.items():
            if src_key in folder_name:
                matched_target = target_val
                break

        if matched_target:
            valid_images = [
                Path(root) / f for f in files
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
            ]
            collected_files[matched_target].extend(valid_images)

    # Perform stratified split (70% train, 15% val, 15% test)
    random.seed(42)
    for cls_name, file_list in collected_files.items():
        total = len(file_list)
        class_counts[cls_name] = total
        if total == 0:
            continue

        random.shuffle(file_list)
        n_train = int(total * 0.70)
        n_val = int(total * 0.15)

        splits = {
            "train": file_list[:n_train],
            "val": file_list[n_train:n_train + n_val],
            "test": file_list[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            dest_dir = processed_dir / split_name / cls_name
            for src_file in split_files:
                shutil.copy2(src_file, dest_dir / src_file.name)

    return class_counts

def report_dataset_status(class_counts: Dict[str, int]):
    """Prints a clear distribution table and highlights missing/supplemental classes."""
    print("\n" + "=" * 70)
    print("📊 Step 3: Disease Class Distribution & Balance Report")
    print("=" * 70)
    print(f"{'Target Schema Class':<25} | {'Images Count':<12} | {'Status'}")
    print("-" * 70)

    for cls_name in TARGET_CLASSES:
        count = class_counts.get(cls_name, 0)
        if cls_name == "Leaf Spot":
            status = "⚠️ MISSING FROM MANGOLEAFBD (Supplemental Roboflow source needed)"
        elif count > 0:
            status = "✅ READY"
        else:
            status = "⏳ Awaiting Ingestion"
        print(f"{cls_name:<25} | {count:<12} | {status}")
    print("-" * 70)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Mango Orchard Health & Pest Intelligence — Phase 1 Disease Ingestion")
    parser.add_argument("--sample", action="store_true", help="Generate sample synthetic dataset for testing")
    args = parser.parse_args()

    print("AI Mango Orchard Health & Pest Intelligence — Phase 1 Disease Ingestion")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check if raw files already exist or download
    existing_files = list(RAW_DIR.glob("**/*.*"))
    download_success = False
    if len(existing_files) < 10:
        download_success = download_mangoleafbd(RAW_DIR)
        if not download_success:
            print("\nℹ️  Kaggle credentials not present.")
            # Check if processed data already exists
            train_dir = PROCESSED_DIR / "train"
            existing_processed = list(train_dir.glob("*/*.*")) if train_dir.exists() else []
            if len(existing_processed) > 0:
                print("Found existing processed datasets.")
            elif args.sample or not check_kaggle_credentials():
                print("Generating structured sample dataset for immediate Phase 1 / Phase 2 verification...")
                try:
                    from ml.datasets.generate_sample_data import generate_disease_dataset
                    generate_disease_dataset(images_per_class=24)
                except Exception as e:
                    print(f"Sample generation notice: {e}")

    # Calculate current class counts across processed dataset
    class_counts = {c: 0 for c in TARGET_CLASSES}
    for split in ["train", "val", "test"]:
        for c in TARGET_CLASSES:
            split_cls_dir = PROCESSED_DIR / split / c
            if split_cls_dir.exists():
                class_counts[c] += len(list(split_cls_dir.glob("*.*")))

    if sum(class_counts.values()) == 0 and len(existing_files) >= 10:
        class_counts = process_and_filter_classes(RAW_DIR, PROCESSED_DIR)

    report_dataset_status(class_counts)

if __name__ == "__main__":
    main()
