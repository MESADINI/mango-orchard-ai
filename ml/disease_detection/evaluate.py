"""
AI Mango Orchard Health & Pest Intelligence — Evaluation of the DEPLOYED disease checkpoint.

Runs the checkpoint exactly as the backend loads it (DiseaseDetector: arch, input size, normalisation and
temperature from the checkpoint) on the group-aware test split and the external set, and writes
results/disease/deployed_eval.json. Refuses to report anything for a missing/untrained checkpoint.
The model comparison itself (CV, statistics, calibration, OOD) is in experiments.py / analysis.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.experiments import external_rows  # noqa: E402
from ml.disease_detection.inference import DISEASE_CLASSES, DiseaseDetector  # noqa: E402
from ml.disease_detection.metrics import classification_metrics, softmax  # noqa: E402
from ml.disease_detection.train import DEFAULT_SAVE_DIR, CHECKPOINT_NAME, RowDataset, read_split, sha256_file  # noqa: E402
from ml.utils.experiment import write_json  # noqa: E402

OUT = REPO_ROOT / "results" / "disease" / "deployed_eval.json"


def evaluate_rows(detector: DiseaseDetector, rows) -> dict:
    loader = DataLoader(RowDataset(rows, detector.transform), batch_size=32, shuffle=False)
    logits = []
    with torch.no_grad():
        for x, _ in loader:
            logits.append(detector.model(x.to(detector.device)).float().cpu().numpy())
    probs = softmax(np.concatenate(logits), detector.temperature)
    y = np.array([r["y"] for r in rows])
    m = classification_metrics(y, probs, detector.classes)
    if detector.ood_threshold is not None:
        m["low_confidence_fraction"] = float((probs.max(1) < detector.ood_threshold).mean())
    return m


def evaluate_model(weights_path: str):
    detector = DiseaseDetector(weights_path=weights_path)
    if not detector.is_trained:
        print(f"No trained weights at {weights_path}; refusing to report metrics for an untrained model.")
        return None
    if detector.classes != DISEASE_CLASSES:
        print(f"Checkpoint class order {detector.classes} differs from {DISEASE_CLASSES}; aborting.")
        return None
    rows = read_split()
    result = {"checkpoint": str(weights_path), "sha256": sha256_file(Path(weights_path)), "arch": detector.arch,
              "temperature": detector.temperature, "ood_threshold": detector.ood_threshold,
              "test": evaluate_rows(detector, [r for r in rows if r["split"] == "test"]),
              "external": evaluate_rows(detector, external_rows())}
    write_json(OUT, result)
    for name in ("test", "external"):
        m = result[name]
        print(f"{name}: n={m['n']} accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
              f"(classes evaluated: {', '.join(m['classes_evaluated'])})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(DEFAULT_SAVE_DIR / CHECKPOINT_NAME))
    evaluate_model(parser.parse_args().weights)
