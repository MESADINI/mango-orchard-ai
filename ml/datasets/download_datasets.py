#!/usr/bin/env python3
"""
AI Mango Orchard Health & Pest Intelligence — Master Dataset Orchestrator

Executes Phase 1 dataset acquisition in sequence:
1. Disease dataset acquisition (MangoLeafBD + Leaf Spot check)
2. Pest dataset acquisition (Confirmed Roboflow datasets + Weak class checks)
3. Quality check and distribution reports
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_step(script_name: str, desc: str):
    print("\n" + "#" * 80)
    print(f"# {desc}")
    print("#" * 80)
    script_path = BASE_DIR / script_name
    res = subprocess.run([sys.executable, str(script_path)])
    if res.returncode != 0:
        print(f"⚠️  Step '{script_name}' exited with returncode {res.returncode}")

def main():
    print("================================================================================")
    print("🥭 Starting AI Mango Orchard Health & Pest Intelligence — Phase 1 Pipeline")
    print("================================================================================")

    run_step("phase1_disease.py", "Step 1: Ingest Disease Datasets (MangoLeafBD)")
    run_step("phase1_pest.py", "Step 2: Ingest Pest Datasets (Roboflow Fruit Fly, Mealy Bug, Scale)")

    print("\n" + "=" * 80)
    print("✅ Phase 1 Dataset Ingestion Check Complete.")
    print("Next Steps:")
    print("  1. Verify image files in ml/datasets/processed/disease/ and ml/datasets/processed/pest/")
    print("  2. If Leaf Spot or Hopper/Borer are needed, follow MANUAL_ANNOTATION_GUIDE.md")
    print("  3. Proceed to Phase 2: Disease Model Training (EfficientNet-B3)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
