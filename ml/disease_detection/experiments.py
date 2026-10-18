#!/usr/bin/env python3
"""
Phase 2 experiment driver (resumable: a run with summary.json is skipped).

    python ml/disease_detection/experiments.py cv        [--archs a,b,...]   5-fold group CV
    python ml/disease_detection/experiments.py final     [--archs a,b,...]   train split -> val (early stop) -> test etc.
    python ml/disease_detection/experiments.py ablation  [--arch A] [--fold 0]                 3 single-fold ablations
    python ml/disease_detection/experiments.py reeval                  recompute every run's logits in fp32 from best.pth

CV fold k: held-out = cv_fold k (evaluation only), early stopping = cv_fold (k+1) mod 5, training = the other three
folds + the always-train groups (cv_fold -2). The test split (cv_fold -1) is never touched by CV.
Default architectures: STUDY_ARCHS (ConvNeXt-T and Swin-T are deferred, not run).

Ablations (reduced scope): one change each to the recipe, trained on exactly the CV fold-0 partition with the same
seed, evaluated on held-out fold 0, and compared with the existing CV run cv/<arch>/fold0 (the baseline). One run per
variant, so differences are single-run estimates, not mean +- SD.

Evaluation sets of the final runs:
  test       group-aware MangoLeafBD test split
  external   independent images (docs/datasets/external.md; 4 classes, no Healthy)
  ood_near   MangoLeafBD leaves of the 3 classes the model does not know (Cutting Weevil, Die Back, Gall Midge)
  ood_far    images from the pest dataset test split (traps, grape, coffee leaves): not mango leaf disease photos
Outputs: results/disease/{cv,final,ablation}/...  (analysed by analysis.py)
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.inference import DISEASE_CLASSES  # noqa: E402
from ml.disease_detection.train import (ARCHS, STUDY_ARCHS, build_transforms, create_model, read_split,  # noqa: E402
                                        run_training, save_eval_logits, select_device, sha256_file)

RESULTS = REPO_ROOT / "results" / "disease"
EXTERNAL_MANIFEST = REPO_ROOT / "ml" / "datasets" / "processed" / "external" / "manifest.csv"
MLBD_MANIFEST = REPO_ROOT / "ml" / "datasets" / "raw" / "mangoleafbd" / "manifest.csv"
PEST_SPLIT = REPO_ROOT / "ml" / "datasets" / "processed" / "pest" / "split.csv"
OOD_NEAR_CLASSES = ("Cutting Weevil", "Die Back", "Gall Midge")
N_OOD_FAR = 300
SEED = 42


def _csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def external_rows():
    return [{"path": r["path"], "abs_path": str(REPO_ROOT / r["path"]), "y": DISEASE_CLASSES.index(r["label"]),
             "group": r["sha256"]} for r in _csv(EXTERNAL_MANIFEST)]


def ood_near_rows():
    seen, rows = set(), []
    for r in _csv(MLBD_MANIFEST):
        if r["source_class"] in OOD_NEAR_CLASSES and r["sha256"] not in seen:
            seen.add(r["sha256"])
            rows.append({"path": r["path"], "abs_path": str(REPO_ROOT / r["path"]), "y": -1,
                         "group": r["source_class"]})
    return rows


def ood_far_rows():
    rows = [r for r in _csv(PEST_SPLIT) if r["split"] == "test"]
    rows = random.Random(SEED).sample(rows, min(N_OOD_FAR, len(rows)))
    return [{"path": r["image"], "abs_path": str(REPO_ROOT / r["image"]), "y": -1, "group": r["source"]}
            for r in rows]


def final_eval_sets(rows):
    return {"test": [r for r in rows if r["split"] == "test"], "external": external_rows(),
            "ood_near": ood_near_rows(), "ood_far": ood_far_rows()}


def _done(out: Path) -> bool:
    return (out / "summary.json").exists()


def cv_partition(rows, k):
    """(train, early_stop, heldout) rows for CV fold k."""
    es = (k + 1) % 5
    train = [r for r in rows if r["cv_fold"] == -2 or (r["cv_fold"] >= 0 and r["cv_fold"] not in (k, es))]
    return train, [r for r in rows if r["cv_fold"] == es], [r for r in rows if r["cv_fold"] == k]


def run_cv(archs, folds=range(5)):
    rows = read_split()
    for arch in archs:
        for k in folds:
            out = RESULTS / "cv" / arch / f"fold{k}"
            if _done(out):
                print(f"skip {out}")
                continue
            train, val, held = cv_partition(rows, k)
            run_training(arch, train, val, out, {"heldout": held}, {"cv_fold": k, "early_stop_fold": (k + 1) % 5})


def run_final(archs):
    rows = read_split()
    sets = final_eval_sets(rows)
    for arch in archs:
        out = RESULTS / "final" / arch
        if _done(out):
            print(f"skip {out}")
            continue
        run_training(arch, [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"],
                     out, sets)


# baseline = the existing CV run cv/<arch>/fold<k> (default recipe, same partition and seed)
ABLATIONS = {
    "no_pretraining": {"pretrained": False},          # pretrained vs scratch
    "basic_augmentation": {"augmentation": "basic"},  # strong vs basic augmentation
    "img_size_256": {"img_size": 256},                # input size 224 -> 256 (288 does not fit in 4 GB at batch 32)
}


def run_ablation(arch, fold=0, variants=None):
    train, val, held = cv_partition(read_split(), fold)
    for name in variants or ABLATIONS:
        out = RESULTS / "ablation" / arch / name / f"fold{fold}"
        if _done(out):
            print(f"skip {out}")
            continue
        run_training(arch, train, val, out, {"heldout": held},
                     {**ABLATIONS[name], "ablation": name, "cv_fold": fold, "early_stop_fold": (fold + 1) % 5})


def run_sets(run_dir: Path, rows) -> dict:
    """The exact evaluation sets a finished run was evaluated on, rebuilt from its location and config."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if "cv_fold" in cfg:                                   # CV runs and the fold-based ablations
        _, val, held = cv_partition(rows, cfg["cv_fold"])
        return {"val": val, "heldout": held}
    return {"val": [r for r in rows if r["split"] == "val"], **final_eval_sets(rows)}


def reevaluate(run_dirs=None, device=None) -> list:
    """Recomputes logits_<set>.npz of finished runs in fp32 from best.pth (verified against summary.json)."""
    import numpy as np
    import torch
    rows = read_split()
    dev = select_device(device)
    run_dirs = run_dirs or sorted(p.parent for p in RESULTS.glob("**/summary.json") if (p.parent / "best.pth").exists())
    done = []
    for run in run_dirs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if sha256_file(run / "best.pth") != summary["checkpoint_sha256"]:
            raise RuntimeError(f"{run}/best.pth does not match its recorded SHA-256")
        ckpt = torch.load(run / "best.pth", map_location=dev)
        model = create_model(ckpt["arch"], pretrained=False).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        _, eval_tf = build_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"], "basic")
        sets = run_sets(run, rows)
        for name, set_rows in sets.items():            # same images in the same order as the original evaluation
            old = run / f"logits_{name}.npz"
            if old.exists():
                assert list(np.load(old)["path"]) == [r.get("path", r["abs_path"]) for r in set_rows], (run, name)
        save_eval_logits(model, sets, eval_tf, dev, run)
        summary["eval_precision"] = "fp32"
        summary["eval_sets"] = {k: len(v) for k, v in sets.items()}
        (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        done.append(str(run.relative_to(REPO_ROOT)))
        print(f"re-evaluated (fp32): {done[-1]}", flush=True)
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["cv", "final", "ablation", "reeval"])
    parser.add_argument("--archs", default="study", help="'study' (" + ",".join(STUDY_ARCHS) + ") or a "
                        "comma-separated subset of " + ",".join(ARCHS))
    parser.add_argument("--arch", default="efficientnet_b3", help="architecture for the ablation")
    parser.add_argument("--fold", type=int, default=0, help="CV fold the ablations are trained/evaluated on")
    parser.add_argument("--variants", default="", help="comma-separated subset of " + ",".join(ABLATIONS))
    parser.add_argument("--folds", default="0,1,2,3,4")
    a = parser.parse_args()
    archs = list(STUDY_ARCHS) if a.archs == "study" else a.archs.split(",")
    if a.what == "cv":
        run_cv(archs, [int(f) for f in a.folds.split(",")])
    elif a.what == "final":
        run_final(archs)
    elif a.what == "ablation":
        run_ablation(a.arch, a.fold, a.variants.split(",") if a.variants else None)
    else:
        reevaluate()
