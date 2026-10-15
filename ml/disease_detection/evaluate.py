"""
AI Mango Orchard Health & Pest Intelligence — Model Evaluation Pipeline
Computes precision, recall, F1-score, and confusion matrix on the test split.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.inference import DISEASE_CLASSES, DiseaseDetector, build_eval_transform
from ml.disease_detection.train import DEFAULT_DATA_DIR, DEFAULT_SAVE_DIR, CHECKPOINT_NAME, load_split


def evaluate_model(data_dir: str, weights_path: str):
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_dir):
        print(f"Test directory {test_dir} not found.")
        return None

    detector = DiseaseDetector(weights_path=weights_path)
    if not detector.is_trained:
        print(f"No trained weights at {weights_path}; refusing to report metrics for an untrained model.")
        return None
    if detector.classes != DISEASE_CLASSES:
        print(f"Checkpoint class order {detector.classes} differs from {DISEASE_CLASSES}; aborting.")
        return None

    test_dataset = load_split(test_dir, build_eval_transform())
    if len(test_dataset) == 0:
        print("Test split is empty.")
        return None
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds = detector.model(images.to(detector.device)).argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    labels = list(range(len(DISEASE_CLASSES)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=DISEASE_CLASSES, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "=" * 60)
    print("EfficientNet-B3 Disease Detection Classification Report")
    print("=" * 60)
    print(report)
    print("Confusion Matrix (rows = true, cols = predicted, order = DISEASE_CLASSES):")
    print(matrix)
    return {"report": report, "confusion_matrix": matrix.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--weights", type=str, default=str(DEFAULT_SAVE_DIR / CHECKPOINT_NAME))
    args = parser.parse_args()
    evaluate_model(args.data_dir, args.weights)
