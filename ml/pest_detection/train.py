#!/usr/bin/env python3
"""
AI Mango Orchard Health & Pest Intelligence — Pest Detection Experiments (Phase 3)

    python ml/pest_detection/train.py train  [--models yolov8n,yolov8s,yolo11n] [--epochs 50]   resumable
    python ml/pest_detection/train.py deploy                                                     copy the selected model

Every run: seed 42, deterministic=True, COCO-pretrained start (weights in ml/pest_detection/pretrained/, SHA-256
recorded), early stopping on the val split (patience 15), then one evaluation on val and on the test split.
Model selection uses val mAP50-95 only; the test split is reported, never used to choose.
Outputs: results/pest/<model>/{config.json, env.json, summary.json, results.csv}; weights stay in the run directory
(not committed) and the selected one is copied, SHA-verified, to ml/pest_detection/weights/pest_detector.pt with
pest_detector.json beside it (class list, host_verified flags, source run, SHA-256).

Data caveat (docs/datasets/pest.md): each class comes from one non-mango source (Fruit Fly: grayscale trap photos;
Mealy Bug: grape/stock photos; Scale Insect: coffee leaves), so class and image domain are confounded. mAP measures
detection within these sources, not on mango orchards; every class is therefore host_verified = false.
Gated classes Mango Hopper and Stem Borer are not trained (< 100 annotations).
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.utils.experiment import start_experiment, write_json  # noqa: E402

CONFIRMED_CLASSES = ["Fruit Fly", "Mealy Bug", "Scale Insect"]
UNRELIABLE_CLASSES = ["Mango Hopper", "Stem Borer"]

DATA_YAML = REPO_ROOT / "ml" / "datasets" / "processed" / "pest" / "data.yaml"
SOURCE_REVIEW = REPO_ROOT / "ml" / "datasets" / "pest_source_review.json"
PRETRAINED_DIR = Path(__file__).resolve().parent / "pretrained"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEPLOYED_WEIGHTS = WEIGHTS_DIR / "pest_detector.pt"
RESULTS = REPO_ROOT / "results" / "pest"

# Official Ultralytics release assets; SHA-256 computed on our download (the release lists no digest)
PRETRAINED = {
    "yolov8n": {"file": "yolov8n.pt", "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
                "sha256": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"},
    "yolov8s": {"file": "yolov8s.pt", "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
                "sha256": "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"},
    "yolo11n": {"file": "yolo11n.pt", "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"},
}
STUDY_MODELS = ("yolov8n", "yolov8s", "yolo11n")
SEED = 42
HPARAMS = {"epochs": 50, "patience": 15, "imgsz": 640, "optimizer": "SGD", "lr0": 0.01, "workers": 2,
           "cos_lr": False, "close_mosaic": 10}
BATCH = {"yolov8n": 16, "yolov8s": 8, "yolo11n": 16}        # 4 GB GPU


def sha256_file(path: Path) -> str:
    from ml.datasets.fetch import sha256_file as _sha
    return _sha(Path(path))


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def host_verified_by_class(review_path: Path = SOURCE_REVIEW) -> Dict[str, bool]:
    """A class is host-verified only if its source's reviewed host is mango (ml/datasets/pest_source_review.json)."""
    review = json.loads(Path(review_path).read_text(encoding="utf-8")) if Path(review_path).exists() else {}
    slug = {c: c.lower().replace(" ", "_") for c in CONFIRMED_CLASSES}
    return {c: review.get(slug[c], {}).get("host") == "mango" for c in CONFIRMED_CLASSES}


def pretrained_path(model: str) -> Path:
    spec = PRETRAINED[model]
    path = PRETRAINED_DIR / spec["file"]
    if not path.exists():
        from ml.datasets.fetch import parallel_download
        parallel_download(spec["url"], path, sha256=spec["sha256"], workers=16, chunk_size=1 << 19)
    if spec["sha256"] and sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"{path} does not match the recorded SHA-256")
    return path


def _metrics(val_result, names: Dict[int, str]) -> dict:
    box = val_result.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {"precision": float(box.p[i]), "recall": float(box.r[i]),
                                     "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i])}
    return {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50),
            "map50_95": float(box.map), "per_class": per_class,
            "speed_ms_per_image": {k: float(v) for k, v in val_result.speed.items()}}


def run_model(model: str, epochs: int = HPARAMS["epochs"], device: Optional[str] = None,
              data_yaml: Path = DATA_YAML, results_dir: Path = RESULTS) -> dict:
    """Trains one detector (resumable: skipped when summary.json exists) and evaluates it on val and test."""
    out_dir = Path(results_dir) / model
    if (out_dir / "summary.json").exists():
        return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    import torch
    from ultralytics import YOLO, settings

    settings.update({"weights_dir": str(PRETRAINED_DIR)})   # AMP self-check loads yolo11n.pt from here, no download
    device = device if device is not None else ("0" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    weights = pretrained_path(model)
    cfg = {"model": model, "pretrained": _rel(weights), "pretrained_sha256": sha256_file(weights),
           "data": _rel(data_yaml), "data_yaml_sha256": sha256_file(data_yaml),
           "seed": SEED, "deterministic": True, "batch": BATCH.get(model, 16), "device": device,
           **HPARAMS, "epochs": epochs, "model_selection": "val mAP50-95 (Ultralytics fitness = 0.1 mAP50 + 0.9 mAP50-95)",
           "host_verified": host_verified_by_class()}
    start_experiment(model, cfg, results_dir=Path(results_dir))

    if (out_dir / "train").exists():                      # an interrupted run restarts from scratch
        shutil.rmtree(out_dir / "train")
    yolo = YOLO(str(weights))
    yolo.train(data=str(data_yaml), epochs=epochs, patience=cfg["patience"], imgsz=cfg["imgsz"], batch=cfg["batch"],
               device=device, seed=SEED, deterministic=True, optimizer=cfg["optimizer"], lr0=cfg["lr0"],
               cos_lr=cfg["cos_lr"], close_mosaic=cfg["close_mosaic"], workers=cfg["workers"],
               project=str(out_dir), name="train", exist_ok=True, plots=False, verbose=False)
    best = out_dir / "train" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished without {best}")
    shutil.copy2(out_dir / "train" / "results.csv", out_dir / "results.csv")

    trained = YOLO(str(best))
    evals = {}
    for split in ("val", "test"):
        r = trained.val(data=str(data_yaml), split=split, imgsz=cfg["imgsz"], batch=cfg["batch"], device=device,
                        conf=0.001, iou=0.7, plots=False, verbose=False, project=str(out_dir), name=f"eval_{split}",
                        exist_ok=True)
        evals[split] = _metrics(r, trained.names)
    with open(out_dir / "results.csv", encoding="utf-8") as f:
        epochs_log = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    summary = {"model": model, "checkpoint": _rel(best), "checkpoint_sha256": sha256_file(best),
               "epochs_run": len(epochs_log), "names": [trained.names[i] for i in sorted(trained.names)],
               "n_params": int(sum(p.numel() for p in trained.model.parameters())),
               "checkpoint_mb": round(best.stat().st_size / 1e6, 2), "val": evals["val"], "test": evals["test"],
               "eval_settings": {"conf": 0.001, "iou": 0.7, "imgsz": cfg["imgsz"]}}
    write_json(out_dir / "summary.json", summary)
    return summary


def select_model(summaries: Dict[str, dict]) -> str:
    """Highest val mAP50-95; ties broken by fewer parameters. The test split plays no part."""
    return max(summaries, key=lambda m: (summaries[m]["val"]["map50_95"], -summaries[m]["n_params"]))


def write_table(summaries: Dict[str, dict], results_dir: Path = RESULTS) -> Path:
    lines = ["| Model | Params | Epochs | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test P / R | "
             + " | ".join(f"Test AP50-95 {c}" for c in CONFIRMED_CLASSES) + " |",
             "|---" * (7 + len(CONFIRMED_CLASSES)) + "|"]
    for m, s in summaries.items():
        te = s["test"]
        per = " | ".join(f"{te['per_class'].get(c, {}).get('ap50_95', float('nan')):.3f}" for c in CONFIRMED_CLASSES)
        lines.append(f"| {m} | {s['n_params'] / 1e6:.1f} M | {s['epochs_run']} | {s['val']['map50_95']:.3f} | "
                     f"{te['map50']:.3f} | {te['map50_95']:.3f} | {te['precision']:.3f} / {te['recall']:.3f} | {per} |")
    lines += ["", "Single run per model (seed 42). Classes are confounded with their source (one non-mango source per "
              "class), so these numbers do not estimate performance on mango orchard images."]
    path = Path(results_dir) / "models.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def deploy(results_dir: Path = RESULTS, weights_dir: Path = WEIGHTS_DIR) -> dict:
    """Copies the selected run's best.pt (by explicit path, never a glob) and verifies the copy's SHA-256."""
    summaries = {m: json.loads((Path(results_dir) / m / "summary.json").read_text(encoding="utf-8"))
                 for m in STUDY_MODELS if (Path(results_dir) / m / "summary.json").exists()}
    if not summaries:
        raise SystemExit("No finished pest runs in results/pest/")
    chosen = select_model(summaries)
    src = REPO_ROOT / summaries[chosen]["checkpoint"]
    if sha256_file(src) != summaries[chosen]["checkpoint_sha256"]:
        raise RuntimeError(f"{src} does not match its recorded SHA-256")
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    dst = weights_dir / DEPLOYED_WEIGHTS.name
    shutil.copyfile(src, dst)
    digest = sha256_file(dst)
    if digest != summaries[chosen]["checkpoint_sha256"]:
        raise RuntimeError("Deployed copy differs from the source checkpoint")
    meta = {"model": chosen, "source_checkpoint": summaries[chosen]["checkpoint"], "sha256": digest,
            "classes": summaries[chosen]["names"], "host_verified": host_verified_by_class(),
            "selection": "highest val mAP50-95", "val_map50_95": summaries[chosen]["val"]["map50_95"],
            "test_map50_95": summaries[chosen]["test"]["map50_95"]}
    write_json(dst.with_suffix(".json"), meta)
    write_json(Path(results_dir) / "deployed.json", meta)
    write_table(summaries, results_dir)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["train", "deploy"])
    parser.add_argument("--models", default=",".join(STUDY_MODELS))
    parser.add_argument("--epochs", type=int, default=HPARAMS["epochs"])
    parser.add_argument("--device", default=None)
    a = parser.parse_args()
    if a.what == "train":
        done = {}
        for m in a.models.split(","):
            print(f"=== {m} ===", flush=True)
            done[m] = run_model(m, a.epochs, a.device)
            print(f"{m}: val mAP50-95 {done[m]['val']['map50_95']:.3f}  test mAP50-95 {done[m]['test']['map50_95']:.3f}",
                  flush=True)
        write_table(done)
    else:
        print(json.dumps(deploy(), indent=2))
