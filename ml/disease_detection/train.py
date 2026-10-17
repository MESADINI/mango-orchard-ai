"""
AI Mango Orchard Health & Pest Intelligence — Disease Classifier Training

Any timm backbone in ARCHS; 5 classes (DISEASE_CLASSES). Data comes from the group-aware split written by
ml/datasets/build_disease_dataset.py (ml/datasets/processed/disease/split.csv).

Training recipe (all switchable for ablations): ImageNet-pretrained weights, strong augmentation (random resized
crop, dihedral flips/rotations, colour jitter, blur, random erasing), label smoothing 0.1, inverse-frequency class
weights, AdamW + linear warm-up + cosine decay, mixed precision (AMP) on CUDA, early stopping on validation macro-F1.
The best checkpoint embeds arch, classes, input size, normalisation and its training config; its SHA-256 is written
next to it. After training, logits for every evaluation set are saved for offline analysis (analysis.py).

Single run (train split -> val split):
    python ml/disease_detection/train.py --arch efficientnet_b3
The full comparison (5-fold group CV, final models, ablations) is driven by experiments.py.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.inference import DISEASE_CLASSES, build_eval_transform, select_device  # noqa: E402
from ml.disease_detection.metrics import classification_metrics  # noqa: E402
from ml.utils.experiment import set_seed, start_experiment, write_json  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "ml" / "datasets" / "processed" / "disease"
SPLIT_CSV = DEFAULT_DATA_DIR / "split.csv"
DEFAULT_SAVE_DIR = REPO_ROOT / "ml" / "disease_detection" / "weights"
CHECKPOINT_NAME = "disease_classifier.pth"  # deployed checkpoint read by the backend (any arch; see deploy.py)

# Short name -> timm model id (ImageNet-1k pretrained weights, pinned tags)
ARCHS = {
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "resnet50": "resnet50.a1_in1k",
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
}
# Architectures in the Phase 2 study. ConvNeXt-T and Swin-T are supported but DEFERRED (not run): too slow on the
# 4 GB GPU for the time budget. Nothing is reported for them.
STUDY_ARCHS = ("efficientnet_b3", "resnet50", "mobilenetv3_large")
DEFERRED_ARCHS = tuple(a for a in ARCHS if a not in STUDY_ARCHS)

DEFAULT_CONFIG = {
    "img_size": 224,
    "epochs": 30,
    "warmup_epochs": 2,
    "batch_size": 32,
    "lr": 2e-4,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "class_weights": True,
    "augmentation": "strong",     # strong | basic
    "pretrained": True,
    "amp": True,
    "patience": 6,
    "seed": 42,
}


# ----------------------------------------------------------------------------- data
def read_split(path: Path = SPLIT_CSV) -> List[dict]:
    """Rows of split.csv with an integer `y` (index into DISEASE_CLASSES) and absolute `abs_path`."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["y"] = DISEASE_CLASSES.index(r["label"])
        r["cv_fold"] = int(r["cv_fold"])
        r["abs_path"] = str(REPO_ROOT / r["path"])
    return rows


def load_split(split_dir: str, transform) -> datasets.ImageFolder:
    """
    Loads an ImageFolder split whose labels are remapped to the canonical DISEASE_CLASSES order.

    ImageFolder assigns indices alphabetically by folder name, which does not match
    DISEASE_CLASSES. Without remapping, a trained model's outputs would be mislabelled
    at inference time.
    """
    dataset = datasets.ImageFolder(split_dir, transform=transform, allow_empty=True)
    unknown = [c for c in dataset.classes if c not in DISEASE_CLASSES]
    if unknown:
        raise ValueError(f"Unknown class folders in {split_dir}: {unknown}. Expected only {DISEASE_CLASSES}.")

    remap = {folder_idx: DISEASE_CLASSES.index(name) for name, folder_idx in dataset.class_to_idx.items()}
    dataset.samples = [(path, remap[idx]) for path, idx in dataset.samples]
    dataset.targets = [idx for _, idx in dataset.samples]
    dataset.classes = list(DISEASE_CLASSES)
    dataset.class_to_idx = {name: i for i, name in enumerate(DISEASE_CLASSES)}
    return dataset


class RowDataset(Dataset):
    """Images from rows ({abs_path, y}); decoded once and kept in memory (the whole set is < 1 GB)."""

    _cache: Dict[str, Image.Image] = {}

    def __init__(self, rows: List[dict], transform):
        self.paths = [r["abs_path"] for r in rows]
        self.y = [int(r.get("y", -1)) for r in rows]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = self._cache.get(p)
        if img is None:
            with Image.open(p) as im:
                img = im.convert("RGB")
                img.thumbnail((640, 640))   # external camera photos are up to 4000 px
            self._cache[p] = img
        return self.transform(img), self.y[i]


class RandomRot90:
    """Rotation by a random multiple of 90 degrees (leaves are photographed at arbitrary orientation)."""

    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)))
        return img.rotate(90 * k, expand=True) if k else img


def build_transforms(img_size: int = 224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                     augmentation: str = "strong"):
    normalize = transforms.Normalize(mean=mean, std=std)
    if augmentation == "strong":
        train = transforms.Compose([
            RandomRot90(),
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    elif augmentation == "basic":
        train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"unknown augmentation {augmentation!r}")
    return train, build_eval_transform(img_size, mean=mean, std=std)


def class_weights(labels: List[int], n_classes: int = len(DISEASE_CLASSES)) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1 over present classes (absent classes get 0)."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    present = counts > 0
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def lr_at(epoch_float: float, cfg: dict) -> float:
    """Linear warm-up then cosine decay to 1% of the base learning rate."""
    if epoch_float < cfg["warmup_epochs"]:
        return cfg["lr"] * epoch_float / cfg["warmup_epochs"]
    t = (epoch_float - cfg["warmup_epochs"]) / max(cfg["epochs"] - cfg["warmup_epochs"], 1)
    return cfg["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- model
PRETRAINED_DIR = REPO_ROOT / "ml" / "disease_detection" / "pretrained"   # git-ignored weight cache


def ensure_pretrained(timm_id: str) -> Path:
    """Downloads timm/<timm_id>/model.safetensors from the Hugging Face Hub with the parallel range downloader
    (a single connection here runs at ~30 KB/s) and verifies it against the Hub's SHA-256 (X-Linked-ETag)."""
    import urllib.request

    from ml.datasets.fetch import USER_AGENT, parallel_download

    dest = PRETRAINED_DIR / timm_id / "model.safetensors"
    if dest.exists():
        return dest
    url = f"https://huggingface.co/timm/{timm_id}/resolve/main/model.safetensors"

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        resp = urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
    except urllib.error.HTTPError as e:   # the 302 to the CDN carries the LFS headers
        resp = e
    sha = (resp.headers.get("X-Linked-ETag") or "").strip('"') or None
    print(f"Downloading pretrained weights {timm_id} (sha256 {sha})", flush=True)
    parallel_download(url, dest, sha256=sha, workers=32)
    return dest


def create_model(arch: str, pretrained: bool = True, num_classes: int = len(DISEASE_CLASSES)) -> nn.Module:
    timm_id = ARCHS.get(arch, arch)
    if not pretrained:
        return timm.create_model(timm_id, pretrained=False, num_classes=num_classes)
    weights = ensure_pretrained(timm_id)
    return timm.create_model(timm_id, pretrained=True, num_classes=num_classes,
                             pretrained_cfg_overlay={"file": str(weights)})


def normalisation(model: nn.Module):
    cfg = timm.data.resolve_data_config({}, model=model)
    return tuple(cfg["mean"]), tuple(cfg["std"])


@torch.no_grad()
def predict_logits(model: nn.Module, rows: List[dict], transform, device, batch_size: int = 64,
                   amp: bool = True) -> np.ndarray:
    model.eval()
    loader = DataLoader(RowDataset(rows, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, len(DISEASE_CLASSES)), np.float32)


# ----------------------------------------------------------------------------- training
def run_training(arch: str, train_rows: List[dict], val_rows: List[dict], out_dir: Path,
                 eval_sets: Optional[Dict[str, List[dict]]] = None, config: Optional[dict] = None,
                 device: Optional[str] = None, num_workers: int = 0) -> dict:
    """Trains `arch` on train_rows with early stopping on val_rows; saves best.pth (+ .sha256), history.csv,
    summary.json and logits_<name>.npz for val and every eval set. Returns the summary."""
    cfg = {**DEFAULT_CONFIG, **(config or {}), "arch": arch, "timm_id": ARCHS.get(arch, arch),
           "n_train": len(train_rows), "n_val": len(val_rows)}
    out_dir = Path(out_dir)
    start_experiment(out_dir.name, cfg, results_dir=out_dir.parent)   # config.json + env.json, seeds
    set_seed(cfg["seed"])
    dev = select_device(device)
    amp = cfg["amp"] and dev.type == "cuda"

    model = create_model(arch, cfg["pretrained"]).to(dev)
    mean, std = normalisation(model)
    train_tf, eval_tf = build_transforms(cfg["img_size"], mean, std, cfg["augmentation"])
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(RowDataset(train_rows, train_tf), batch_size=cfg["batch_size"], shuffle=True,
                              drop_last=len(train_rows) > cfg["batch_size"], generator=g, num_workers=num_workers,
                              persistent_workers=num_workers > 0)
    weight = class_weights([r["y"] for r in train_rows]).to(dev) if cfg["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    y_val = np.array([r["y"] for r in val_rows], dtype=np.int64)

    best, best_epoch, bad, history = None, -1, 0, []
    ckpt_path = out_dir / "best.pth"
    t0 = time.time()
    steps = len(train_loader)
    for epoch in range(cfg["epochs"]):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(epoch + step / steps, cfg)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            n_seen += len(y)

        logits = predict_logits(model, val_rows, eval_tf, dev, amp=amp)
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y_val)))
        m = classification_metrics(y_val, logits, DISEASE_CLASSES)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / max(n_seen, 1), "val_loss": val_loss,
               "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "lr": optimizer.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{arch} {out_dir.name}] epoch {epoch + 1:02d} loss {row['train_loss']:.4f} val_loss {val_loss:.4f} "
              f"val_acc {m['accuracy']:.4f} val_f1 {m['macro_f1']:.4f} ({row['seconds']}s)", flush=True)
        key = (m["macro_f1"], -val_loss)
        if best is None or key > best:
            best, best_epoch, bad = key, epoch + 1, 0
            torch.save({
                "state_dict": model.state_dict(), "classes": list(DISEASE_CLASSES), "arch": arch,
                "timm_id": cfg["timm_id"], "img_size": cfg["img_size"], "mean": mean, "std": std,
                "epoch": epoch + 1, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"], "config": cfg,
            }, ckpt_path)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"[{arch} {out_dir.name}] early stop at epoch {epoch + 1} (best {best_epoch})", flush=True)
                break

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    model.load_state_dict(torch.load(ckpt_path, map_location=dev)["state_dict"])
    sets = {"val": val_rows, **(eval_sets or {})}
    save_eval_logits(model, sets, eval_tf, dev, out_dir)
    digest = sha256_file(ckpt_path)
    (out_dir / "best.pth.sha256").write_text(f"{digest}  best.pth\n", encoding="ascii")
    summary = {"arch": arch, "best_epoch": best_epoch, "epochs_run": len(history),
               "best_val_macro_f1": best[0], "train_seconds": round(time.time() - t0, 1),
               "checkpoint_sha256": digest, "checkpoint_bytes": ckpt_path.stat().st_size,
               "n_params": int(sum(p.numel() for p in model.parameters())),
               "eval_sets": {k: len(v) for k, v in sets.items()}, "eval_precision": "fp32"}
    write_json(out_dir / "summary.json", summary)
    return summary


def save_eval_logits(model: nn.Module, sets: Dict[str, List[dict]], eval_tf, device, out_dir: Path) -> None:
    """Writes logits_<name>.npz for each set. Always fp32: under fp16 autocast EfficientNet-B3 overflowed to
    non-finite logits on some out-of-distribution images, so reported numbers never use AMP."""
    for name, rows in sets.items():
        logits = predict_logits(model, rows, eval_tf, device, amp=False)
        np.savez_compressed(Path(out_dir) / f"logits_{name}.npz", logits=logits,
                            y=np.array([int(r.get("y", -1)) for r in rows]),
                            path=np.array([r.get("path", r["abs_path"]) for r in rows]),
                            group=np.array([str(r.get("group", "")) for r in rows]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="efficientnet_b3", choices=sorted(ARCHS))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "disease" / "single"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--deploy", action="store_true",
                        help=f"Copy the best checkpoint to {DEFAULT_SAVE_DIR / CHECKPOINT_NAME}")
    a = parser.parse_args()
    if not SPLIT_CSV.exists():
        sys.exit(f"{SPLIT_CSV} not found. Run: python ml/datasets/build_disease_dataset.py")
    rows = read_split()
    out = Path(a.out) / a.arch
    s = run_training(a.arch, [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"],
                     out, {"test": [r for r in rows if r["split"] == "test"]},
                     {"epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr, "pretrained": a.pretrained})
    print(json.dumps(s, indent=2))
    if a.deploy:
        import shutil
        DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "best.pth", DEFAULT_SAVE_DIR / CHECKPOINT_NAME)
        print(f"Deployed to {DEFAULT_SAVE_DIR / CHECKPOINT_NAME}")
