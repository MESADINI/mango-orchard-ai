#!/usr/bin/env python3
"""
Validates the colour-based lesion segmenter (severity.py) against hand-drawn masks.

    python ml/severity/validate.py sample   [--n 60]          writes the stratified list of images to annotate
    python ml/severity/validate.py convert  DIR               LabelMe JSON polygons ("leaf", "lesion") -> mask PNGs
    python ml/severity/validate.py evaluate [--masks DIR]     -> results/severity/validation.json + per_image.csv

Mask format (docs/annotation/SEVERITY_MASKS.md): one PNG per image, same stem, grey levels
0 = background, 128 = healthy leaf, 255 = lesion. Severity = lesion / (healthy leaf + lesion).
Metrics: lesion IoU and Dice, leaf IoU, severity MAE (percentage points) and Spearman rho, per class and overall,
with 95% bootstrap CIs over images. Refuses to report anything with fewer than MIN_MASKS masks.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.inference import DISEASE_CLASSES  # noqa: E402
from ml.severity.severity import MAX_SIDE, compute_lesion_severity  # noqa: E402
from ml.utils.experiment import git_info, write_json  # noqa: E402

SPLIT_CSV = REPO_ROOT / "ml" / "datasets" / "processed" / "disease" / "split.csv"
SAMPLE_CSV = REPO_ROOT / "ml" / "datasets" / "annotation" / "severity_sample.csv"
MASK_DIR = REPO_ROOT / "ml" / "datasets" / "annotation" / "severity_masks"
OUT_DIR = REPO_ROOT / "results" / "severity"
MIN_MASKS = 50
BACKGROUND, LEAF, LESION = 0, 128, 255
SEED = 42


def make_sample(n: int = 60, split_csv: Path = SPLIT_CSV, out: Path = SAMPLE_CSV, seed: int = SEED):
    """Stratified sample of test-split images, one image per leaf group, equal share per class."""
    with open(split_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    rng = random.Random(seed)
    per_class = {c: [] for c in DISEASE_CLASSES}
    for r in rows:
        per_class[r["label"]].append(r)
    picked = []
    quota = {c: n // len(DISEASE_CLASSES) + (i < n % len(DISEASE_CLASSES)) for i, c in enumerate(DISEASE_CLASSES)}
    for c, items in per_class.items():
        rng.shuffle(items)
        seen = set()
        for r in items:
            if r["group"] not in seen and len([p for p in picked if p["label"] == c]) < quota[c]:
                seen.add(r["group"])
                picked.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "mask_file"])
        for r in picked:
            w.writerow([r["path"], r["label"], r["group"], Path(r["path"]).stem + ".png"])
    return picked


def labelme_to_mask(json_path: Path, out_dir: Path) -> Path:
    """Rasterises LabelMe polygons: 'leaf' -> 128, then 'lesion' -> 255 on top."""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    mask = np.zeros((d["imageHeight"], d["imageWidth"]), np.uint8)
    for label, value in (("leaf", LEAF), ("lesion", LESION)):
        for s in d["shapes"]:
            if s["label"].strip().lower() == label:
                cv2.fillPoly(mask, [np.round(np.array(s["points"])).astype(np.int32)], value)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(json_path).stem + ".png")
    Image.fromarray(mask).save(out)
    return out


def read_mask(path: Path, shape) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    if m.shape != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    q = np.full(m.shape, BACKGROUND, np.uint8)       # snap anti-aliased edges to the nearest level
    q[(m >= 64) & (m < 192)] = LEAF
    q[m >= 192] = LESION
    return q


def _iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def _dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def _spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else None


def score_image(image_rgb: np.ndarray, gt: np.ndarray) -> dict:
    pred = compute_lesion_severity(image_rgb, return_masks=True)
    shape = pred["leaf_mask"].shape                  # working resolution (longest side <= MAX_SIDE)
    gt = cv2.resize(gt, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_leaf, gt_lesion = gt != BACKGROUND, gt == LESION
    gt_sev = 100.0 * gt_lesion.sum() / gt_leaf.sum() if gt_leaf.sum() else 0.0
    return {"lesion_iou": _iou(pred["lesion_mask"], gt_lesion), "lesion_dice": _dice(pred["lesion_mask"], gt_lesion),
            "leaf_iou": _iou(pred["leaf_mask"], gt_leaf), "severity_true": gt_sev,
            "severity_pred": pred["severity_percentage"], "abs_error": abs(pred["severity_percentage"] - gt_sev)}


def _summ(rows, seed=SEED, n_boot=2000):
    keys = ("lesion_iou", "lesion_dice", "leaf_iou", "abs_error")
    rng = np.random.default_rng(seed)
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows])
        boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
        out[k if k != "abs_error" else "severity_mae_pp"] = {
            "mean": float(v.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    out["severity_spearman"] = _spearman([r["severity_true"] for r in rows], [r["severity_pred"] for r in rows])
    return out


def evaluate(mask_dir: Path = MASK_DIR, sample_csv: Path = SAMPLE_CSV, out_dir: Path = OUT_DIR,
             min_masks: int = MIN_MASKS):
    with open(sample_csv, encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    have = [r for r in sample if (mask_dir / r["mask_file"]).exists()]
    if len(have) < min_masks:
        print(f"Only {len(have)} of the required {min_masks} masks are in {mask_dir}; not reporting metrics. "
              "See docs/annotation/SEVERITY_MASKS.md.")
        return None
    per_image = []
    for r in have:
        img = np.array(Image.open(REPO_ROOT / r["path"]).convert("RGB"))
        gt = read_mask(mask_dir / r["mask_file"], img.shape[:2])
        per_image.append({"path": r["path"], "label": r["label"], **score_image(img, gt)})
    result = {"overall": _summ(per_image),
              "per_class": {c: _summ([p for p in per_image if p["label"] == c])
                            for c in DISEASE_CLASSES if any(p["label"] == c for p in per_image)},
              "n_masks": len(per_image), "working_resolution_max_side": MAX_SIDE, "git": git_info()}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "validation.json", result)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0]))
        w.writeheader()
        w.writerows(per_image)
    o = result["overall"]
    lines = [f"Segmenter vs expert masks, n = {o['n']} (results/severity/validation.json; bootstrap 95% CI).\n",
             "| Metric | Mean [95% CI] |", "|---|---|"]
    for k in ("lesion_iou", "lesion_dice", "leaf_iou", "severity_mae_pp"):
        lines.append(f"| {k} | {o[k]['mean']:.3f} [{o[k]['ci95'][0]:.3f}, {o[k]['ci95'][1]:.3f}] |")
    rho = o["severity_spearman"]
    lines.append(f"| severity Spearman | {'n/a' if rho is None else f'{rho:.3f}'} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=60)
    c = sub.add_parser("convert")
    c.add_argument("dir", type=Path)
    e = sub.add_parser("evaluate")
    e.add_argument("--masks", type=Path, default=MASK_DIR)
    a = ap.parse_args()
    if a.cmd == "sample":
        print(f"{len(make_sample(a.n))} images -> {SAMPLE_CSV}")
    elif a.cmd == "convert":
        for j in sorted(a.dir.glob("*.json")):
            print(labelme_to_mask(j, MASK_DIR))
    else:
        evaluate(a.masks)
