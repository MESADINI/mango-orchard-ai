#!/usr/bin/env python3
"""
Builds the leakage-free disease dataset from the downloaded MangoLeafBD release.

Steps
  1. Read ml/datasets/raw/mangoleafbd/manifest.csv (written by download_mangoleafbd.py).
  2. Drop exact byte duplicates (same SHA-256).
  3. Group near-duplicates across ALL 8 source classes (pHash + DINOv2 retrieval + SIFT/RANSAC verification,
     see dedup.py).
     A group whose images carry different labels is a label conflict and is excluded.
  4. Keep the target classes (MangoLeafBD has no Leaf Spot; see docs/datasets/mangoleafbd.md).
  5. Group-aware, class-stratified split: 70% train / 15% val / 15% test, plus 5 group-aware
     cross-validation folds over train+val (column `cv_fold`: 0-4; -1 = test, never in CV;
     -2 = oversized group, always in the training part of every fold).
  6. Materialise ImageFolder trees ml/datasets/processed/disease/{train,val,test}/<Class>/ and
     ml/datasets/processed/disease/split.csv.
Evidence: results/data/disease_dataset.json
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets import dedup  # noqa: E402
from ml.disease_detection.inference import DISEASE_CLASSES  # noqa: E402
from ml.utils.experiment import collect_env, set_seed, write_json  # noqa: E402

RAW_DIR = REPO_ROOT / "ml" / "datasets" / "raw" / "mangoleafbd"
OUT_DIR = REPO_ROOT / "ml" / "datasets" / "processed" / "disease"
CACHE = RAW_DIR / "dedup_features.npz"
EVIDENCE = REPO_ROOT / "results" / "data" / "disease_dataset.json"

# MangoLeafBD label -> project label (None = excluded: pest damage / stem symptom, not a foliar disease)
CLASS_MAP = {
    "Anthracnose": "Anthracnose",
    "Bacterial Canker": "Bacterial Canker",
    "Powdery Mildew": "Powdery Mildew",
    "Sooty Mould": "Sooty Mold",
    "Healthy": "Healthy",
    "Cutting Weevil": None,
    "Die Back": None,
    "Gall Midge": None,
}

# Chosen with --calibrate (evidence: results/data/disease_dedup_calibration.json, docs/phases/PHASE_1.md):
# with one-to-one SIFT matching no cross-class pair is ever linked (0 / 4,196 candidates at k=40, even at 10 inliers);
# k=40 is converged (k=80 changes the group count by < 1%); 12 inliers is where linked pairs are visually almost all
# the same leaf, and it yields ~1,780 groups, matching the authors' "around 1800 distinct leaves".
PHASH_THRESHOLD = 0
CANDIDATE_K = 40
INLIER_THRESHOLD = 12
SEED = 42
CALIBRATION = REPO_ROOT / "results" / "data" / "disease_dedup_calibration.json"
CALIBRATION_SHEETS = REPO_ROOT / "results" / "data" / "dedup_review"


def read_manifest(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique_by_sha(rows):
    seen = {}
    for r in rows:
        seen.setdefault(r["sha256"], r)
    return list(seen.values())


def load_or_compute_features(rows, cache: Path = CACHE):
    shas = [r["sha256"] for r in rows]
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if list(data["sha256"]) == shas:
            print(f"Using cached dedup features: {cache}")
            return data["phash"], data["emb"]
    paths = [REPO_ROOT / r["path"] for r in rows]
    print(f"Computing pHash for {len(paths)} images...")
    bits = dedup.phash_bits(paths)
    print(f"Computing DINOv2 embeddings for {len(paths)} images x 8 views on {dedup.embedding_device()}...")
    emb = dedup.dino_embeddings(paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, sha256=np.array(shas), phash=bits, emb=emb)
    return bits, emb


def geometric_links(rows, emb, k=CANDIDATE_K):
    """DINOv2 top-k candidate pairs and their SIFT/RANSAC inlier counts."""
    cos = dedup.dihedral_max_cosine(emb)
    pairs = dedup.candidate_pairs(cos, k)
    print(f"Verifying {len(pairs)} candidate pairs (top-{k}) with SIFT + RANSAC...")
    feats = dedup.sift_features([REPO_ROOT / r["path"] for r in rows])
    return cos, pairs, dedup.verify_pairs(feats, pairs)


def threshold_sweep(ph, pairs, inliers, classes, phash_threshold, thresholds):
    """Per inlier threshold: groups, links, cross-class links/groups (cross-class links are false: two images with
    different labels should not be the same leaf, so their rate estimates the false-link rate) and largest group."""
    classes = np.asarray(classes)
    cross = classes[pairs[:, 0]] != classes[pairs[:, 1]]
    out = []
    for th in thresholds:
        g = dedup.group_near_duplicates(ph, pairs, inliers, phash_threshold, th)
        labels_by_group = defaultdict(set)
        for c, gid in zip(classes, g["groups"]):
            labels_by_group[gid].add(c)
        conflict = {k for k, v in labels_by_group.items() if len(v) > 1}
        sizes = Counter(g["groups"].tolist())
        n_cross_links = int((inliers[cross] >= th).sum())
        out.append({
            "inlier_threshold": int(th),
            "n_groups": int(g["groups"].max() + 1),
            "n_verified_links": g["n_verified_edges"],
            "n_cross_class_candidates": int(cross.sum()),
            "n_cross_class_links": n_cross_links,
            "false_link_rate_estimate": round(n_cross_links / max(int(cross.sum()), 1), 6),
            "n_cross_class_conflict_groups": len(conflict),
            "n_images_in_conflict_groups": int(sum(sizes[k] for k in conflict)),
            "largest_group": int(max(sizes.values())),
        })
    return out


def pair_sheet(pairs, rows, cos, out: Path, cell: int = 150):
    """Side-by-side thumbnails of image pairs with their cosine, for visual threshold calibration."""
    from PIL import Image, ImageDraw

    if not pairs:
        return
    cols = 4  # 4 pairs per row
    sheet = Image.new("RGB", (cols * 2 * cell, ((len(pairs) + cols - 1) // cols) * (cell + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (a, b) in enumerate(pairs):
        x0, y0 = (k % cols) * 2 * cell, (k // cols) * (cell + 16)
        for off, idx in ((0, a), (cell, b)):
            with Image.open(REPO_ROOT / rows[idx]["path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((cell - 4, cell - 4))
                sheet.paste(im, (x0 + off + 2, y0 + 2))
        draw.text((x0 + 4, y0 + cell), f"{cos[a, b]:.3f} {rows[a]['source_class'][:10]}|{rows[b]['source_class'][:10]}",
                  fill=(200, 0, 0) if rows[a]["source_class"] != rows[b]["source_class"] else (0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def calibrate(phash_threshold=PHASH_THRESHOLD):
    """Writes the evidence for choosing CANDIDATE_K and INLIER_THRESHOLD (and for rejecting a cosine threshold):
    cosine-band and inlier-band pair contact sheets, and a threshold sweep for several k. Builds nothing."""
    rows = unique_by_sha(read_manifest(RAW_DIR / "manifest.csv"))
    bits, emb = load_or_compute_features(rows)
    ph = dedup.phash_min_distance(bits)
    cos = dedup.dihedral_max_cosine(emb)
    cls = np.asarray([r["source_class"] for r in rows])
    rng = np.random.default_rng(SEED)

    def sheets(pairs, values, bands, prefix, matrix):
        info = []
        for lo, hi in bands:
            sel = np.where((values >= lo) & (values < hi))[0]
            cross = cls[pairs[sel, 0]] != cls[pairs[sel, 1]]
            name = f"{lo:g}-{hi:g}"
            for kind, s in (("same", sel[~cross]), ("cross", sel[cross])):
                pick = rng.choice(s, min(16, len(s)), replace=False) if len(s) else []
                pair_sheet([tuple(pairs[k]) for k in pick], rows, matrix,
                           CALIBRATION_SHEETS / f"{prefix}_{name}_{kind}.jpg")
            info.append({"band": name, "n_pairs_same_class": int((~cross).sum()),
                         "n_pairs_cross_class": int(cross.sum())})
        return info

    iu, ju = np.triu_indices(len(rows), 1)
    cos_bands = sheets(np.stack([iu, ju], axis=1), cos[iu, ju],
                       [(0.80, 0.84), (0.84, 0.87), (0.87, 0.90), (0.90, 0.93), (0.93, 0.96), (0.96, 1.01)],
                       "cos", cos)
    feats = dedup.sift_features([REPO_ROOT / r["path"] for r in rows])
    sweeps, inl_bands = {}, []
    for k in (10, 20, 40, 80):
        pairs = dedup.candidate_pairs(cos, k)
        inliers = dedup.verify_pairs(feats, pairs)
        sweeps[str(k)] = threshold_sweep(ph, pairs, inliers, cls, phash_threshold, [10, 12, 15, 18, 20, 25, 30, 40])
        if k == CANDIDATE_K:
            m = np.zeros((len(rows), len(rows)), np.float32)
            m[pairs[:, 0], pairs[:, 1]] = m[pairs[:, 1], pairs[:, 0]] = inliers
            inl_bands = sheets(pairs, inliers, [(8, 12), (12, 18), (18, 25), (25, 40), (40, 10000)], "inliers", m)
    result = {
        "phash_threshold": phash_threshold, "n_images": len(rows), "embedding_device": dedup.embedding_device(),
        "chosen": {"candidate_k": CANDIDATE_K, "inlier_threshold": INLIER_THRESHOLD},
        "cosine_bands_all_pairs": cos_bands, "inlier_bands_at_chosen_k": inl_bands,
        "sweep_by_candidate_k": sweeps,
        "authors_statement": "around 1800 [of 4000] are of distinct leaves (Mendeley record hxsnvwty3r)",
        "sheets": "results/data/dedup_review/{cos,inliers}_<band>_<same|cross>.jpg (red caption = cross-class; "
                  "caption number = cosine or inlier count)",
    }
    write_json(CALIBRATION, result)
    for k, sw in sweeps.items():
        for s in sw:
            print(f"k={k}", s)
    return result


ALWAYS_TRAIN = -2  # cv_fold value of oversized groups: in the training part of every CV fold, never held out


def oversized_groups(labels, groups, unit_fraction=0.05):
    """Groups larger than one 5% split unit of their class: one leaf copied that often cannot be placed in val/test
    without dominating that class's metrics (MangoLeafBD has a single Sooty Mould leaf with 189 copies)."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    class_size = Counter(labels.tolist())
    sizes = Counter(groups.tolist())
    label_of = dict(zip(groups.tolist(), labels.tolist()))
    return {g for g, n in sizes.items() if n > unit_fraction * class_size[label_of[g]]}


def stratified_group_split(labels, groups, seed=SEED):
    """
    Returns (split, cv_fold) arrays. Oversized groups (see oversized_groups) go to train with cv_fold ALWAYS_TRAIN.
    The rest uses StratifiedGroupKFold with 20 folds as 5% units: 3 units -> test, 3 -> val, 14 -> train; then
    5 group folds over the remaining train+val for CV. cv_fold is -1 for test.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    labels, groups = np.asarray(labels), np.asarray(groups)
    forced = np.isin(groups, list(oversized_groups(labels, groups)))
    split = np.full(len(labels), "train", dtype=object)
    cv_fold = np.full(len(labels), ALWAYS_TRAIN, dtype=int)
    rest = np.where(~forced)[0]
    unit = np.empty(len(rest), dtype=int)
    for k, (_, idx) in enumerate(StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
                                 .split(np.zeros(len(rest)), labels[rest], groups[rest])):
        unit[idx] = k
    split[rest] = np.where(unit < 3, "test", np.where(unit < 6, "val", "train"))
    cv_fold[rest[split[rest] == "test"]] = -1
    pool = rest[split[rest] != "test"]
    for k, (_, idx) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                                 .split(np.zeros(len(pool)), labels[pool], groups[pool])):
        cv_fold[pool[idx]] = k
    return split.astype(str), cv_fold


def materialise(records, out_dir: Path):
    for split in ("train", "val", "test"):
        shutil.rmtree(out_dir / split, ignore_errors=True)
    for r in records:
        dest = out_dir / r["split"] / r["label"] / Path(r["source_path"]).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / r["source_path"], dest)
        r["path"] = dest.relative_to(REPO_ROOT).as_posix()
    with open(out_dir / "split.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["path", "label", "source_class", "sha256", "group", "split", "cv_fold", "source_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in records)


def main(phash_threshold=PHASH_THRESHOLD, candidate_k=CANDIDATE_K, inlier_threshold=INLIER_THRESHOLD):
    set_seed(SEED)
    target_classes = [c for c in DISEASE_CLASSES]
    unknown = {v for v in CLASS_MAP.values() if v} - set(target_classes)
    if unknown:
        raise ValueError(f"CLASS_MAP targets {unknown} not in DISEASE_CLASSES {target_classes}")

    all_rows = read_manifest(RAW_DIR / "manifest.csv")
    rows = unique_by_sha(all_rows)
    n_exact_dups = len(all_rows) - len(rows)

    bits, emb = load_or_compute_features(rows)
    ph = dedup.phash_min_distance(bits)
    _, pairs, inliers = geometric_links(rows, emb, candidate_k)
    grouping = dedup.group_near_duplicates(ph, pairs, inliers, phash_threshold, inlier_threshold)
    groups = grouping["groups"]

    labels_by_group = defaultdict(set)
    for r, g in zip(rows, groups):
        labels_by_group[g].add(r["source_class"])
    conflict_groups = {g for g, ls in labels_by_group.items() if len(ls) > 1}

    records, excluded = [], Counter()
    for r, g in zip(rows, groups):
        target = CLASS_MAP[r["source_class"]]
        if target is None:
            excluded["non_target_class"] += 1
        elif g in conflict_groups:
            excluded["cross_class_conflict"] += 1
        else:
            records.append({"source_path": r["path"], "source_class": r["source_class"], "label": target,
                            "sha256": r["sha256"], "group": int(g)})

    split, cv_fold = stratified_group_split([r["label"] for r in records], [r["group"] for r in records])
    for r, s, f in zip(records, split, cv_fold):
        r["split"], r["cv_fold"] = str(s), int(f)
    materialise(records, OUT_DIR)

    def count(key_fn):
        c = defaultdict(Counter)
        for r in records:
            c[r["split"]][key_fn(r)] += 1
        return {s: dict(sorted(c[s].items())) for s in ("train", "val", "test")}

    groups_per_split = defaultdict(set)
    for r in records:
        groups_per_split[r["split"]].add(r["group"])
    overlap = {f"{a}&{b}": len(groups_per_split[a] & groups_per_split[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}

    all_group_sizes = Counter(Counter(groups.tolist()).values())
    evidence = {
        "source": "MangoLeafBD (doi:10.17632/hxsnvwty3r.1), CC BY-NC 3.0",
        "classes": target_classes,
        "class_map": CLASS_MAP,
        "dedup": {
            "method": "union-find over pHash distance (dihedral) and DINOv2 top-k candidates verified by "
                      "SIFT + RANSAC similarity-transform inliers",
            "dino_model": dedup.DINO_MODEL,
            "embedding_device": dedup.embedding_device(),
            "phash_threshold": phash_threshold,
            "candidate_k": candidate_k,
            "inlier_threshold": inlier_threshold,
            "n_candidate_pairs": int(len(pairs)),
            "n_source_images": len(all_rows),
            "n_exact_duplicates_removed": n_exact_dups,
            "n_unique_images": len(rows),
            "n_groups_all_classes": int(groups.max() + 1),
            "n_images_in_multi_image_groups": int(sum(size * n for size, n in all_group_sizes.items() if size > 1)),
            "group_size_histogram": {str(k): v for k, v in sorted(all_group_sizes.items())},
            "n_phash_edges": grouping["n_phash_edges"],
            "n_verified_edges": grouping["n_verified_edges"],
            "largest_groups": [
                {"size": n, "classes": sorted({r["source_class"] for r, g in zip(rows, groups) if g == gid})}
                for gid, n in Counter(groups.tolist()).most_common(5)],
            "n_cross_class_conflict_groups": len(conflict_groups),
        },
        "excluded_images": dict(excluded),
        "n_images": len(records),
        "n_groups": len({r["group"] for r in records}),
        "images_per_split_class": count(lambda r: r["label"]),
        "groups_per_split": {s: len(g) for s, g in groups_per_split.items()},
        "groups_per_class": {c: len({r["group"] for r in records if r["label"] == c}) for c in target_classes},
        "group_overlap_between_splits": overlap,
        "cv_fold_sizes": dict(sorted(Counter(r["cv_fold"] for r in records if r["cv_fold"] >= 0).items())),
        "always_train": {
            "rule": "groups larger than 5% of their class go to train and are never held out in CV (cv_fold -2)",
            "n_groups": len({r["group"] for r in records if r["cv_fold"] == ALWAYS_TRAIN}),
            "n_images_per_class": dict(sorted(Counter(r["label"] for r in records
                                                      if r["cv_fold"] == ALWAYS_TRAIN).items())),
        },
        "seed": SEED,
        "env": collect_env(),
    }
    write_json(EVIDENCE, evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "env"}, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--inlier-threshold", type=int, default=INLIER_THRESHOLD)
    parser.add_argument("--calibrate", action="store_true",
                        help="Only write the threshold sweep and pair contact sheets (results/data/)")
    a = parser.parse_args()
    if a.calibrate:
        calibrate(a.phash_threshold)
    else:
        main(a.phash_threshold, a.candidate_k, a.inlier_threshold)
