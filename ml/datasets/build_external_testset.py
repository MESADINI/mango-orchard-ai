#!/usr/bin/env python3
"""
Builds an independent external test set for cross-dataset generalisation.

Source: "Mango Leaf Disease Dataset" (Rahman, Hasan & Mojumdar, 2024), Mendeley Data doi:10.17632/7ghdbftp54.1,
CC BY 4.0. Photographed in Kashinathpur (Pabna) and Changao (Savar), Bangladesh, Oct-Nov 2024 - different
orchards, devices and year from MangoLeafBD. The release is a zip holding two inner zips, "process data.zip"
(originals) and "Augmented Image.zip"; only the non-augmented images are used, and only the classes the model knows.

Provenance finding (docs/datasets/external.md): the release is NOT fully independent. Most of its images have exactly
MangoLeafBD's pixel sizes (240x320, 320x240, 240x240, ...) and, where checked, are MangoLeafBD leaves with the
background blacked out. Two exclusion rules therefore apply:
  1. Format rule: images whose size is one of MangoLeafBD's sizes are excluded as probable derivatives.
  2. Leakage guard: every remaining candidate is compared with every MangoLeafBD image using the disease-dataset
     criteria (exact SHA-256, dihedral pHash distance 0, or a top-k DINOv2 neighbour confirmed by SIFT + RANSAC).
Excluded images are listed with their reason in ml/datasets/processed/external/excluded.csv.

Outputs: ml/datasets/processed/external/<Class>/*.jpg, ml/datasets/processed/external/manifest.csv
Evidence: results/data/external_testset.json
"""

import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets import dedup  # noqa: E402
from ml.datasets.build_disease_dataset import (  # noqa: E402
    CACHE as MLBD_CACHE, CANDIDATE_K, INLIER_THRESHOLD, PHASH_THRESHOLD, RAW_DIR as MLBD_RAW, read_manifest,
)
from ml.datasets.fetch import parallel_download  # noqa: E402
from ml.disease_detection.inference import DISEASE_CLASSES  # noqa: E402
from ml.utils.experiment import collect_env, write_json  # noqa: E402

URL = ("https://data.mendeley.com/public-files/datasets/7ghdbftp54/files/"
       "a1995507-daa3-49cd-a923-267bfa82d437/file_downloaded")
SHA256 = "1ccf1520499dd61efc71fd24d1af5928af865b1a542a8e818eae344965539249"
LICENSE = "CC BY 4.0"
CITATION = ("Rahman, M.S., Hasan, R., Mojumdar, M.U. (2024). Mango Leaf Disease Dataset. Mendeley Data, V1. "
            "doi:10.17632/7ghdbftp54.1")

RAW_DIR = REPO_ROOT / "ml" / "datasets" / "raw" / "external_7ghdbftp54"
OUT_DIR = REPO_ROOT / "ml" / "datasets" / "processed" / "external"
EVIDENCE = REPO_ROOT / "results" / "data" / "external_testset.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# normalised folder name -> project class (None = class the model does not know)
FOLDER_MAP = {
    "anthracnose": "Anthracnose",
    "bacterialcanker": "Bacterial Canker",
    "powderymildew": "Powdery Mildew",
    "sootymould": "Sooty Mold",
    "sootymold": "Sooty Mold",
    "healthy": "Healthy",
    "cuttingweevil": None,
    "dieback": None,
    "gallmidge": None,
}


def normalise(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def classify_member(member: str):
    """Returns (class_or_None, is_augmented) for a path inside the zip, or (None, None) if not an image."""
    p = Path(member)
    if p.suffix.lower() not in IMAGE_EXTS:
        return None, None
    parts = [normalise(x) for x in p.parts[:-1]]
    augmented = any("augment" in x for x in parts)
    for part in reversed(parts):
        if part in FOLDER_MAP:
            return FOLDER_MAP[part], augmented
    return None, augmented


def iter_images(zf: zipfile.ZipFile):
    """Yields (member_path, bytes) for every file, descending into inner .zip archives; inner members are named
    '<inner zip path>/<member>' so that classify_member sees the inner zip's name (e.g. 'Augmented Image.zip')."""
    for member in zf.namelist():
        if member.endswith("/"):
            continue
        if member.lower().endswith(".zip"):
            with zipfile.ZipFile(zf.open(member)) as inner:
                for name, data in iter_images(inner):
                    yield f"{member}/{name}", data
        else:
            yield member, zf.read(member)


def main():
    zip_path = parallel_download(URL, RAW_DIR / "source.zip", sha256=SHA256)
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    candidates, skipped = [], Counter()
    with zipfile.ZipFile(zip_path) as zf:
        for member, data in iter_images(zf):
            cls, augmented = classify_member(member)
            if augmented is None:
                skipped["not_an_image"] += 1
                continue
            if augmented:
                skipped["augmented"] += 1
                continue
            if cls is None:
                skipped["class_not_in_model"] += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            dest = RAW_DIR / "extracted" / cls / f"{digest[:16]}{Path(member).suffix.lower()}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            candidates.append({"member": member, "label": cls, "sha256": digest, "raw": dest})

    unique = {}
    for c in candidates:
        unique.setdefault(c["sha256"], c)
    n_exact_dups = len(candidates) - len(unique)
    candidates = list(unique.values())

    # Leakage guard against every MangoLeafBD image
    mlbd = np.load(MLBD_CACHE, allow_pickle=False)
    paths = [c["raw"] for c in candidates]
    bits = dedup.phash_bits(paths)
    emb = dedup.dino_embeddings(paths)
    ph = np.full(len(paths), 64)
    for t in range(8):
        ph = np.minimum(ph, (bits[:, None, 0, :] != mlbd["phash"][None, :, t, :]).sum(-1).min(1))
    sim = np.full((len(paths), len(mlbd["emb"])), -1.0, dtype=np.float32)
    for t in range(8):
        sim = np.maximum(sim, emb[:, 0, :] @ mlbd["emb"][:, t, :].T)
    cos = sim.max(1)
    mlbd_sha = set(mlbd["sha256"].tolist())
    # geometric verification against each candidate's top-k MangoLeafBD neighbours
    path_by_sha = {r["sha256"]: REPO_ROOT / r["path"] for r in read_manifest(MLBD_RAW / "manifest.csv")}
    nn = np.argsort(-sim, axis=1)[:, :CANDIDATE_K]
    needed = np.unique(nn)
    mlbd_feats = dict(zip(needed.tolist(), dedup.sift_features([path_by_sha[mlbd["sha256"][j]] for j in needed])))
    ext_feats = dedup.sift_features(paths)
    inl_all = np.array([[dedup.geometric_inliers(ext_feats[i], mlbd_feats[j]) for j in nn[i]]
                        for i in range(len(paths))], dtype=int)
    inl = inl_all.max(1)
    best_match = [mlbd["sha256"][nn[i][inl_all[i].argmax()]] for i in range(len(paths))]
    mlbd_sizes = {(int(r["width"]), int(r["height"])) for r in read_manifest(MLBD_RAW / "manifest.csv")}
    sizes = []
    for pth in paths:
        with Image.open(pth) as im:
            sizes.append(im.size)

    rows = []
    for c, d, s, n, bm, size in zip(candidates, ph, cos, inl, best_match, sizes):
        near = bool(c["sha256"] in mlbd_sha or d <= PHASH_THRESHOLD or n >= INLIER_THRESHOLD)
        mlbd_format = size in mlbd_sizes
        row = {"label": c["label"], "sha256": c["sha256"], "source_member": c["member"],
               "width": size[0], "height": size[1], "mangoleafbd_format": mlbd_format,
               "min_phash_to_training": int(d), "max_cosine_to_training": round(float(s), 4),
               "max_inliers_to_training": int(n), "best_matching_training_sha256": bm,
               "near_duplicate_of_training": near, "path": ""}
        row["excluded_reason"] = ";".join(r for r, f in (("mangoleafbd_format", mlbd_format),
                                                          ("near_duplicate_of_training", near)) if f)
        if not row["excluded_reason"]:
            dest = OUT_DIR / c["label"] / c["raw"].name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c["raw"], dest)
            row["path"] = dest.relative_to(REPO_ROOT).as_posix()
        rows.append(row)
    kept = [r for r in rows if not r["excluded_reason"]]
    excluded = [r for r in rows if r["excluded_reason"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in (("manifest.csv", kept), ("excluded.csv", excluded)):
        with open(OUT_DIR / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    fmt = [r for r in rows if r["mangoleafbd_format"]]

    evidence = {
        "source": {"doi": "10.17632/7ghdbftp54.1", "url": URL, "sha256_verified": SHA256,
                   "license": LICENSE, "citation": CITATION},
        "classes_in_model": DISEASE_CLASSES,
        "skipped": dict(skipped),
        "n_exact_duplicates_removed": n_exact_dups,
        "n_candidates": len(rows),
        "n_flagged_near_duplicate_of_training": sum(r["near_duplicate_of_training"] for r in rows),
        "n_mangoleafbd_format": len(fmt),
        "n_mangoleafbd_format_confirmed_by_guard": sum(r["near_duplicate_of_training"] for r in fmt),
        "n_native_format": len(rows) - len(fmt),
        "n_native_format_flagged_by_guard": sum(r["near_duplicate_of_training"] for r in rows
                                                if not r["mangoleafbd_format"]),
        "candidates_per_class_and_format": {
            f"{lab}|{'mangoleafbd_format' if m else 'native'}": n
            for (lab, m), n in sorted(Counter((r["label"], r["mangoleafbd_format"]) for r in rows).items())},
        "n_images": len(kept),
        "per_class": dict(sorted(Counter(r["label"] for r in kept).items())),
        "classes_missing": [c for c in DISEASE_CLASSES if c not in {r["label"] for r in kept}],
        "leakage_criteria": {"phash_threshold": PHASH_THRESHOLD, "candidate_k": CANDIDATE_K,
                             "inlier_threshold": INLIER_THRESHOLD},
        "max_inliers_to_training_quantiles": {
            q: int(np.quantile(inl, q)) for q in (0.5, 0.9, 0.99, 1.0)} if len(inl) else {},
        "max_cosine_to_training_quantiles": {
            q: round(float(np.quantile(cos, q)), 4) for q in (0.5, 0.9, 0.99, 1.0)} if len(cos) else {},
        "env": collect_env(),
    }
    write_json(EVIDENCE, evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "env"}, indent=2))


if __name__ == "__main__":
    main()
