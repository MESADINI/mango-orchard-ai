"""
AI Mango Orchard Health & Pest Intelligence — Pest Detection Training
Architecture: YOLOv8 (Ultralytics)
Confirmed Classes: Fruit Fly, Mealy Bug, Scale Insect
Gated Weak Classes: Mango Hopper, Stem Borer (excluded until >= 100 annotations exist)
"""

import os
import shutil
import argparse
from pathlib import Path
import torch

CONFIRMED_CLASSES = ["Fruit Fly", "Mealy Bug", "Scale Insect"]
UNRELIABLE_CLASSES = ["Mango Hopper", "Stem Borer"]

def train_yolo(
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    model_variant: str = "yolov8n.pt",
    batch: int = 16,
    device: str = None
):
    print("=" * 70)
    print("🦗 Training YOLOv8 Pest Detector")
    print(f"Confirmed Classes: {CONFIRMED_CLASSES}")
    print(f"Excluded Gated Classes: {UNRELIABLE_CLASSES} (Requires >= 100 annotated samples)")
    print("=" * 70)

    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        # Fallback to standard relative path if needed
        alt_path = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "pest" / "data.yaml"
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset config {data_yaml} not found. Run `python ml/datasets/build_pest_dataset.py all --include-non-mango` first.")

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"
    print(f"Using compute device: {device}")

    from ultralytics import YOLO
    model = YOLO(model_variant)

    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name="mango_pest_yolov8",
        project="runs/detect",
        exist_ok=True
    )

    # Copy best weights to canonical location
    candidates = list(Path("runs/detect").glob("**/weights/best.pt")) + list(Path("runs/detect").glob("**/weights/last.pt"))
    target_best = weights_dir / "yolov8_best.pt"
    if candidates and candidates[0].exists():
        shutil.copy(candidates[0], target_best)
        print(f"⭐ Saved weights from {candidates[0]} to canonical path: {target_best}")

    print("✅ Pest detection training finished successfully.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml/datasets/processed/pest/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-epoch run to verify pipeline")
    args = parser.parse_args()

    if args.smoke_test:
        print("⚡ Smoke test mode activated: 2 epochs, batch 8")
        args.epochs = 2
        args.batch = 8
        args.model = "yolov8n.pt"

    train_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_variant=args.model,
        batch=args.batch,
        device=args.device
    )
