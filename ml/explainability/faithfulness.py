#!/usr/bin/env python3
"""
Quantitative Grad-CAM evaluation (Phase 4). Runs on CPU by default.

    python ml/explainability/faithfulness.py [--limit N] [--steps 20] [--device cpu]

Deletion / insertion (Petsiuk et al., 2018, RISE): pixels are ranked by saliency and, in `steps` equal chunks,
replaced by a blurred copy of the image (deletion) or copied from the image onto the blurred copy (insertion).
The curve is the softmax probability of the predicted class (deployed temperature); AUC uses the trapezoidal rule
over the fraction of pixels changed. Lower deletion AUC and higher insertion AUC = more faithful.
A saliency map is only meaningful against baselines, so every image is also scored with
  random   a uniformly random pixel ranking (seeded per image)
  centre   a centred Gaussian (leaves are photographed centrally; tests whether Grad-CAM beats a centre prior)
and Grad-CAM is compared with each by a paired Wilcoxon signed-rank test over images.

Pointing game (Zhang et al., 2018): hit if the Grad-CAM maximum lies within 15 px (at 224 px) of an annotated
lesion pixel. Needs the expert lesion masks of ml/severity/validate.py (ml/datasets/annotation/severity_masks/);
skipped with a note when fewer than MIN_MASKS exist.

Images: the MangoLeafBD test split (n = 239), deployed classifier, target = the predicted class.
Outputs: results/xai/{faithfulness.json, faithfulness.md, curves.png}
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.utils.experiment import collect_env, write_json  # noqa: E402

RESULTS = REPO_ROOT / "results" / "xai"
WEIGHTS = REPO_ROOT / "ml" / "disease_detection" / "weights" / "disease_classifier.pth"
SPLIT_CSV = REPO_ROOT / "ml" / "datasets" / "processed" / "disease" / "split.csv"
MASK_DIR = REPO_ROOT / "ml" / "datasets" / "annotation" / "severity_masks"
MIN_MASKS = 50
POINTING_TOLERANCE_PX = 15
SEED = 42
METHODS = ("gradcam", "random", "centre")


def auc(curve: np.ndarray) -> float:
    """Area under a curve sampled at equally spaced fractions of [0, 1] (trapezoidal rule)."""
    curve = np.asarray(curve, dtype=float)
    return float(np.trapz(curve, dx=1.0 / (len(curve) - 1)))


def ranking(saliency: np.ndarray) -> np.ndarray:
    """Pixel indices (flattened) from most to least salient; stable, so ties keep raster order."""
    return np.argsort(-saliency.ravel(), kind="stable")


def perturbation_batch(image: np.ndarray, baseline: np.ndarray, order: np.ndarray, steps: int,
                       mode: str) -> np.ndarray:
    """steps + 1 images (HWC float). deletion: image -> baseline; insertion: baseline -> image, in `order`."""
    h, w = image.shape[:2]
    src, dst = (image, baseline) if mode == "deletion" else (baseline, image)
    out = np.empty((steps + 1, h, w, image.shape[2]), dtype=np.float32)
    flat_src, flat_dst = src.reshape(-1, src.shape[2]), dst.reshape(-1, dst.shape[2])
    for k in range(steps + 1):
        n = int(round(k / steps * h * w))
        cur = flat_src.copy()
        cur[order[:n]] = flat_dst[order[:n]]
        out[k] = cur.reshape(h, w, -1)
    return out


def centre_prior(size: int, sigma_frac: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    c, s = (size - 1) / 2, sigma_frac * size
    return np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * s ** 2))


def pointing_hit(heatmap: np.ndarray, lesion: np.ndarray, tolerance: int = POINTING_TOLERANCE_PX) -> Optional[bool]:
    """True if the heatmap maximum is within `tolerance` px of a lesion pixel; None if the mask has no lesion."""
    if not lesion.any():
        return None
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    ly, lx = np.nonzero(lesion)
    return bool(np.min((ly - y) ** 2 + (lx - x) ** 2) <= tolerance ** 2)


def _test_rows() -> List[dict]:
    with open(SPLIT_CSV, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def run(limit: Optional[int] = None, steps: int = 20, device: str = "cpu", weights: Path = WEIGHTS) -> Dict:
    import cv2
    import torch
    from PIL import Image
    from ml.disease_detection.inference import DiseaseDetector
    from ml.explainability.gradcam import HookBasedGradCAM

    torch.set_num_threads(4)
    det = DiseaseDetector(str(weights), device=device)
    if not det.is_trained:
        raise SystemExit(f"No deployed classifier at {weights}")
    model, T = det.model.eval(), float(det.temperature)
    target_layer = getattr(model, "conv_head", None)
    if target_layer is None:
        raise SystemExit("Grad-CAM target layer conv_head not found (expected an EfficientNet checkpoint)")
    tf = det.transform
    size = tf.transforms[0].size[0]
    mean = np.array(tf.transforms[2].mean, dtype=np.float32)
    std = np.array(tf.transforms[2].std, dtype=np.float32)
    cam = HookBasedGradCAM(model, target_layer)
    prior = centre_prior(size)
    rows = _test_rows()[:limit] if limit else _test_rows()

    def probs(batch_hwc: np.ndarray, cls: int) -> np.ndarray:
        x = torch.from_numpy(((batch_hwc - mean) / std).transpose(0, 3, 1, 2).copy()).to(det.device)
        with torch.no_grad():
            return torch.softmax(model(x).float() / T, dim=1)[:, cls].cpu().numpy()

    per_image, pointing = [], []
    for i, r in enumerate(rows):
        pil = Image.open(REPO_ROOT / r["path"]).convert("RGB")
        img = np.asarray(pil.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy(((img - mean) / std).transpose(2, 0, 1)[None].copy()).to(det.device).requires_grad_(True)
        heat, cls = cam.generate_heatmap(x)
        heat = cv2.resize(heat.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=size / 20)
        rng = np.random.default_rng(SEED + i)
        maps = {"gradcam": heat, "random": rng.random((size, size)), "centre": prior}
        rec = {"image": r["path"], "label": r["label"], "predicted": det.classes[cls]}
        for m, sal in maps.items():
            order = ranking(sal)
            for mode in ("deletion", "insertion"):
                curve = probs(perturbation_batch(img, blurred, order, steps, mode), cls)
                rec[f"{m}_{mode}_auc"] = auc(curve)
                rec[f"{m}_{mode}_curve"] = [round(float(v), 5) for v in curve]
        per_image.append(rec)

        mask = MASK_DIR / (Path(r["path"]).stem + ".png")
        if mask.exists():
            from ml.severity.validate import LESION, read_mask
            gt = read_mask(mask, (size, size)) == LESION
            hit = pointing_hit(heat, gt)
            if hit is not None:
                pointing.append({"image": r["path"], "hit": hit})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} images", flush=True)
    cam.remove_hooks()
    return {"per_image": per_image, "pointing": pointing, "steps": steps, "temperature": T,
            "weights_sha256": _sha(weights)}


def _sha(path: Path) -> str:
    from ml.datasets.fetch import sha256_file
    return sha256_file(Path(path))


def summarise(raw: Dict) -> Dict:
    from scipy.stats import wilcoxon
    rows = raw["per_image"]
    out = {"n_images": len(rows), "steps": raw["steps"], "baseline_fill": "Gaussian blur (sigma = size / 20)"}
    for mode in ("deletion", "insertion"):
        out[mode] = {m: {"mean_auc": float(np.mean([r[f"{m}_{mode}_auc"] for r in rows])),
                         "sd_auc": float(np.std([r[f"{m}_{mode}_auc"] for r in rows]))} for m in METHODS}
        g = np.array([r[f"gradcam_{mode}_auc"] for r in rows])
        for base in ("random", "centre"):
            b = np.array([r[f"{base}_{mode}_auc"] for r in rows])
            better = (g < b) if mode == "deletion" else (g > b)
            diff = g - b
            p = float(wilcoxon(g, b).pvalue) if len(rows) > 1 and np.any(diff != 0) else float("nan")
            out[mode][f"gradcam_vs_{base}"] = {"mean_difference": float(diff.mean()),
                                              "fraction_gradcam_better": float(better.mean()),
                                              "wilcoxon_p": p}
    hits = [p["hit"] for p in raw["pointing"]]
    out["pointing_game"] = ({"n": len(hits), "accuracy": float(np.mean(hits)), "tolerance_px": POINTING_TOLERANCE_PX}
                            if len(hits) >= MIN_MASKS else
                            {"status": f"not run: {len(hits)} annotated lesion masks, {MIN_MASKS} required "
                                       "(docs/annotation/SEVERITY_MASKS.md)"})
    return out


def write_outputs(raw: Dict, summary: Dict, out_dir: Path = RESULTS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "faithfulness.json", {"summary": summary, "temperature": raw["temperature"],
                                                "weights_sha256": raw["weights_sha256"], "seed": SEED,
                                                "per_image": raw["per_image"], "env": collect_env()})
    md = ["# Grad-CAM faithfulness\n",
          f"Deployed classifier, MangoLeafBD test split (n = {summary['n_images']}), target = predicted class, "
          f"{summary['steps']} steps, removed pixels filled with a blurred copy. Lower deletion AUC / higher insertion "
          "AUC = more faithful. Wilcoxon signed-rank over images.\n",
          "| Curve | Grad-CAM | Random | Centre prior | Grad-CAM better than random (p) | Grad-CAM better than centre (p) |",
          "|---|---|---|---|---|---|"]
    for mode in ("deletion", "insertion"):
        s = summary[mode]
        md.append(f"| {mode} AUC | {s['gradcam']['mean_auc']:.3f} ± {s['gradcam']['sd_auc']:.3f} | "
                  f"{s['random']['mean_auc']:.3f} ± {s['random']['sd_auc']:.3f} | "
                  f"{s['centre']['mean_auc']:.3f} ± {s['centre']['sd_auc']:.3f} | "
                  f"{s['gradcam_vs_random']['fraction_gradcam_better']:.2f} ({s['gradcam_vs_random']['wilcoxon_p']:.2g}) | "
                  f"{s['gradcam_vs_centre']['fraction_gradcam_better']:.2f} ({s['gradcam_vs_centre']['wilcoxon_p']:.2g}) |")
    pg = summary["pointing_game"]
    md.append("\nPointing game: " + (f"accuracy {pg['accuracy']:.3f} on {pg['n']} masks (tolerance {pg['tolerance_px']} px)"
                                    if "accuracy" in pg else pg["status"]) + "\n")
    (out_dir / "faithfulness.md").write_text("\n".join(md), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.linspace(0, 1, raw["steps"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, mode in zip(axes, ("deletion", "insertion")):
        for m in METHODS:
            ax.plot(xs, np.mean([r[f"{m}_{mode}_curve"] for r in raw["per_image"]], axis=0), label=m)
        ax.set_title(mode)
        ax.set_xlabel("fraction of pixels changed")
        ax.set_ylabel("p(predicted class)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    raw = run(a.limit, a.steps, a.device)
    summary = summarise(raw)
    write_outputs(raw, summary)
    print((RESULTS / "faithfulness.md").read_text(encoding="utf-8"))
