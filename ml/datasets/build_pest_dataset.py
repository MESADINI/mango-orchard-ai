#!/usr/bin/env python3
"""
Builds the merged YOLO pest dataset from the confirmed Roboflow Universe sources.

Requires the environment variable ROBOFLOW_API_KEY (free Roboflow account). No extra packages.

Steps
  1. download   Each source in SOURCES is exported in YOLOv8 format to ml/datasets/raw/pest/<slug>/.
  2. review     A contact sheet of random images per source is written to results/data/pest_review/<slug>.jpg.
                A human must look at it and record the host plant in ml/datasets/pest_source_review.json
                (`"host": "mango" | "other:<plant>" | "unknown"`). Sources not reviewed as mango are excluded
                unless --include-non-mango is passed (and the exclusion/inclusion is recorded in the evidence).
  3. merge      Labels are remapped to CONFIRMED_CLASSES indices; labels for any other source class are dropped.
                Gated classes (Rule 3: Mango Hopper, Stem Borer) can never enter the training set.
  4. dedup      Roboflow exports contain augmented copies named "<stem>_<ext>.rf.<hash>.jpg". Images are grouped
                by original stem and by dihedral pHash distance 0; exact SHA-256 duplicates are removed.
  5. split      Group-aware, class-stratified 70/15/15 (Roboflow's own splits are ignored: they may leak).
  6. write      ml/datasets/processed/pest/{images,labels}/{train,val,test}/ and data.yaml.
Evidence: results/data/pest_dataset.json
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.utils.experiment import collect_env, set_seed, write_json  # noqa: E402

CONFIRMED_CLASSES = ["Fruit Fly", "Mealy Bug", "Scale Insect"]
UNRELIABLE_CLASSES = ["Mango Hopper", "Stem Borer"]  # Rule 3: never trained until >= 100 verified boxes

# source_classes: Roboflow class names (case-insensitive) that map to `target`; all other classes are dropped.
SOURCES = [
    {"slug": "fruit_fly", "workspace": "school-vakkx", "project": "mango-fruit-fly", "version": 4,
     "target": "Fruit Fly", "source_classes": ["Fruit fly"]},  # 'Pest' is unspecified -> dropped
    {"slug": "mealy_bug", "workspace": "grape-n5yol", "project": "mealybug-b8wfq", "version": 8,
     "target": "Mealy Bug", "source_classes": ["Mealybug"]},  # 'Grape', 'Leaf' dropped
    {"slug": "scale_insect", "workspace": "ralph-matthew-r-aquino", "project": "hemispherical-scale-insect",
     "version": 1, "target": "Scale Insect", "source_classes": ["Hemispherical Scale Insect"]},  # rust dropped
]

RAW_DIR = REPO_ROOT / "ml" / "datasets" / "raw" / "pest"
OUT_DIR = REPO_ROOT / "ml" / "datasets" / "processed" / "pest"
REVIEW_FILE = REPO_ROOT / "ml" / "datasets" / "pest_source_review.json"
REVIEW_SHEETS = REPO_ROOT / "results" / "data" / "pest_review"
EVIDENCE = REPO_ROOT / "results" / "data" / "pest_dataset.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

_RF_SUFFIX = re.compile(r"_(jpe?g|png|bmp|webp)\.rf\.[0-9a-f]+$", re.IGNORECASE)


# ----------------------------------------------------------------------------- pure helpers (unit-tested)
def original_stem(filename: str) -> str:
    """'IMG_12_jpg.rf.ab12cd.jpg' -> 'IMG_12' (Roboflow augmentation/export suffix removed)."""
    return _RF_SUFFIX.sub("", Path(filename).stem)


def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Parses one YOLO line; returns (cls, x, y, w, h) or None if it is invalid (non-numeric, out of range, empty box).
    Accepts boxes (`cls x y w h`) and polygons (`cls x1 y1 x2 y2 ...`, >= 3 points, as in Roboflow segmentation
    exports); a polygon is converted to its enclosing box.
    """
    parts = line.split()
    try:
        cls = int(parts[0])
        vals = [float(v) for v in parts[1:]]
    except (ValueError, IndexError):
        return None
    if cls < 0 or not all(0.0 <= v <= 1.0 for v in vals):
        return None
    if len(vals) == 4:
        x, y, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        x, y = min(xs) + w / 2, min(ys) + h / 2
    else:
        return None
    if w <= 0 or h <= 0:
        return None
    return cls, x, y, w, h


def remap_labels(lines: List[str], id_map: Dict[int, int]) -> Tuple[List[str], Counter]:
    """Keeps only labels whose source id is in id_map, rewritten with the target id. Returns (lines, stats)."""
    out, stats = [], Counter()
    for line in lines:
        if not line.strip():
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            stats["invalid"] += 1
            continue
        cls, x, y, w, h = parsed
        if cls not in id_map:
            stats["dropped_other_class"] += 1
            continue
        out.append(f"{id_map[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        stats["kept"] += 1
    return out, stats


def build_id_map(source_names: List[str], target: str, source_classes: Optional[List[str]]) -> Dict[int, int]:
    """Maps Roboflow class indices to the CONFIRMED_CLASSES index of `target`."""
    if target in UNRELIABLE_CLASSES or target not in CONFIRMED_CLASSES:
        raise ValueError(f"Target {target!r} is not a confirmed class (Rule 3 gating)")
    wanted = {s.lower() for s in source_classes} if source_classes else None
    target_id = CONFIRMED_CLASSES.index(target)
    id_map = {}
    for i, name in enumerate(source_names):
        if any(g.lower() in name.lower() for g in UNRELIABLE_CLASSES):
            continue
        if wanted is None or name.lower() in wanted:
            id_map[i] = target_id
    return id_map


# ----------------------------------------------------------------------------- pipeline steps
def export_link(workspace: str, project: str, version: int, api_key: str, fmt: str = "yolov8",
                 poll_seconds: int = 10, max_polls: int = 60) -> str:
    """
    Asks the Roboflow REST API for a dataset export link (the export is generated on first request).
    The roboflow pip SDK is deliberately not used: it would force numpy 2 / opencv 5 or old certifi.
    """
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}?api_key={api_key}"
    for _ in range(max_polls):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mango-orchard-ai/1.0"}),
                                        timeout=120) as r:
                info = json.loads(r.read())
        except Exception as e:  # never let the URL (which contains the API key) reach logs or tracebacks
            raise RuntimeError(f"Roboflow API request for {workspace}/{project}/{version} failed: "
                               f"{type(e).__name__} {getattr(e, 'code', '')}".strip()) from None
        link = (info.get("export") or {}).get("link")
        if link:
            return link
        time.sleep(poll_seconds)  # export still being generated
    raise TimeoutError(f"Roboflow export for {workspace}/{project}/{version} not ready")


def download(sources) -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is not set. Create a free Roboflow account, copy the private API key "
                         "(Settings -> API Keys) and run:  export ROBOFLOW_API_KEY=...")
    for s in sources:
        dest = RAW_DIR / s["slug"]
        marker = dest / "SOURCE_SHA256.txt"  # written last: its presence means the extraction is complete
        if marker.exists():
            print(f"{s['slug']}: already downloaded and extracted")
            continue
        zip_path = RAW_DIR / f"{s['slug']}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not _zip_ok(zip_path):
            print(f"Downloading {s['workspace']}/{s['project']} v{s['version']} ...")
            link = export_link(s["workspace"], s["project"], s["version"], api_key)
            tmp = zip_path.with_suffix(".part")
            with urllib.request.urlopen(link, timeout=600) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(zip_path)
            if not _zip_ok(zip_path):
                raise IOError(f"{zip_path.name} is not a valid zip archive")
        else:
            print(f"{s['slug']}: re-extracting existing intact {zip_path.name}")
        shutil.rmtree(dest, ignore_errors=True)  # discard any partial extraction
        extract_short_names(zip_path, dest)
        marker.write_text(hashlib.sha256(zip_path.read_bytes()).hexdigest() + "\n")


def short_name(member_name: str) -> str:
    """
    Deterministic short file name (Windows paths are limited to 260 characters and some Roboflow exports keep
    very long original web filenames). An image and its label share a stem, so they get the same short stem.
    """
    p = Path(member_name)
    full_stem = p.stem
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", original_stem(p.name))[:40]
    return f"{prefix}_{hashlib.sha1(full_stem.encode('utf-8')).hexdigest()[:10]}{p.suffix.lower()}"


def extract_short_names(zip_path: Path, dest: Path) -> None:
    """Extracts a Roboflow export, shortening image/label file names; writes names.csv (short -> original)."""
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.parent.name in ("images", "labels"):
                rel = member.parent / short_name(member.name)
                rows.append({"short": rel.as_posix(), "original": info.filename})
            else:
                rel = member
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(info))
    with open(dest / "names.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["short", "original"])
        writer.writeheader()
        writer.writerows(rows)


def _zip_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def read_source(slug: str):
    """
    Returns (class_names, [(image_path, label_path_or_None, original_filename), ...]) for an export extracted
    by extract_short_names(). original_filename is the name inside the Roboflow zip (used for stem grouping).
    """
    import yaml

    root = RAW_DIR / slug
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    with open(root / "names.csv", encoding="utf-8") as f:
        original = {r["short"]: Path(r["original"]).name for r in csv.DictReader(f)}
    items = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() in IMAGE_EXTS and img.parent.name == "images":
            label = img.parent.parent / "labels" / f"{img.stem}.txt"
            rel = img.relative_to(root).as_posix()
            items.append((img, label if label.exists() else None, original.get(rel, img.name)))
    return list(names), items


def review() -> dict:
    """
    Writes, per source, a contact sheet with the ground-truth boxes drawn (results/data/pest_review/<slug>.jpg)
    and a summary of label counts per source class (results/data/pest_review/summary.json), for the
    host-plant review. Builds nothing.
    """
    from PIL import Image, ImageDraw

    summary = {}
    for s in SOURCES:
        names, items = read_source(s["slug"])
        boxes_per_class, images_per_class = Counter(), Counter()
        for _, label, _ in items:
            classes_here = set()
            for line in (label.read_text(encoding="utf-8").splitlines() if label else []):
                parsed = parse_label_line(line)
                if parsed and parsed[0] < len(names):
                    boxes_per_class[names[parsed[0]]] += 1
                    classes_here.add(names[parsed[0]])
            images_per_class.update(classes_here)
        rng = random.Random(SEED)
        chosen = rng.sample(items, min(24, len(items)))
        cell, cols = 220, 6
        sheet = Image.new("RGB", (cols * cell, ((len(chosen) + cols - 1) // cols) * cell), "white")
        for k, (img_path, label, _) in enumerate(chosen):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                draw = ImageDraw.Draw(im)
                w, h = im.size
                for line in (label.read_text(encoding="utf-8").splitlines() if label else []):
                    parsed = parse_label_line(line)
                    if parsed:
                        c, x, y, bw, bh = parsed
                        draw.rectangle([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h],
                                       outline=(255, 0, 0) if c == 0 else (0, 120, 255), width=max(2, w // 150))
                im.thumbnail((cell, cell))
                sheet.paste(im, ((k % cols) * cell, (k // cols) * cell))
        REVIEW_SHEETS.mkdir(parents=True, exist_ok=True)
        sheet.save(REVIEW_SHEETS / f"{s['slug']}.jpg", quality=85)
        summary[s["slug"]] = {
            "roboflow": f"{s['workspace']}/{s['project']}/v{s['version']}",
            "source_class_names": names,
            "n_images": len(items),
            "boxes_per_source_class": dict(boxes_per_class),
            "images_per_source_class": dict(images_per_class),
            "images_without_labels": sum(1 for _, lbl, _ in items if lbl is None or not lbl.read_text().strip()),
            "contact_sheet": f"results/data/pest_review/{s['slug']}.jpg (red = class 0, blue = other classes)",
        }
    write_json(REVIEW_SHEETS / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def load_review() -> Dict[str, dict]:
    if REVIEW_FILE.exists():
        return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    return {}


def group_ids(records) -> np.ndarray:
    """Union of (same source + same original stem) and (dihedral pHash distance 0)."""
    from ml.datasets import dedup

    edges = []
    by_stem = defaultdict(list)
    for i, r in enumerate(records):
        by_stem[(r["source"], r["stem"])].append(i)
    for idx in by_stem.values():
        edges += [(idx[0], j) for j in idx[1:]]
    bits = dedup.phash_bits([r["image"] for r in records])
    dist = dedup.phash_min_distance(bits)
    i, j = np.where(np.triu(dist == 0, 1))
    edges += list(zip(i.tolist(), j.tolist()))
    return dedup.union_find_groups(len(records), np.array(edges, dtype=int).reshape(-1, 2))


def split_groups(labels, groups, seed=SEED):
    from sklearn.model_selection import StratifiedGroupKFold

    unit = np.empty(len(labels), dtype=int)
    for k, (_, idx) in enumerate(StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
                                 .split(np.zeros(len(labels)), labels, groups)):
        unit[idx] = k
    return np.where(unit < 3, "test", np.where(unit < 6, "val", "train"))


def build(include_non_mango: bool = False) -> dict:
    set_seed(SEED)
    review = load_review()
    records, source_report, excluded_sources = [], {}, {}
    seen_sha = set()
    for s in SOURCES:
        names, items = read_source(s["slug"])
        host = review.get(s["slug"], {}).get("host", "unknown")
        info = {"roboflow": f"{s['workspace']}/{s['project']}/v{s['version']}", "source_class_names": names,
                "target": s["target"], "n_images_in_export": len(items), "host_review": review.get(s["slug"])}
        info["status"] = "mango" if host == "mango" else f"FLAGGED proxy data (host {host!r}, not mango)"
        if host != "mango" and not include_non_mango:
            excluded_sources[s["slug"]] = f"host review is {host!r} (needs 'mango')"
            source_report[s["slug"]] = info
            continue
        id_map = build_id_map(names, s["target"], s["source_classes"])
        stats, dup, only_other = Counter(), 0, 0
        for img, label, orig_name in items:
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            if digest in seen_sha:
                dup += 1
                continue
            seen_sha.add(digest)
            lines = label.read_text(encoding="utf-8").splitlines() if label else []
            kept, st = remap_labels(lines, id_map)
            stats.update(st)
            if not kept and st["dropped_other_class"]:
                # every box was a dropped class (e.g. 'Grape'); target pests in it may be unlabelled -> unsafe
                only_other += 1
                continue
            records.append({"source": s["slug"], "image": img, "stem": original_stem(orig_name),
                            "labels": kept, "target": s["target"], "sha256": digest})
        info.update({"label_stats": dict(stats), "exact_duplicates_removed": dup,
                     "images_dropped_only_other_classes": only_other, "id_map": id_map})
        source_report[s["slug"]] = info

    if not records:
        raise SystemExit(f"No source passed the host review. Excluded: {excluded_sources}. "
                         f"Review the contact sheets in {REVIEW_SHEETS} and fill {REVIEW_FILE}.")

    groups = group_ids(records)
    split = split_groups([r["target"] for r in records], groups)

    for sub in ("images", "labels"):
        shutil.rmtree(OUT_DIR / sub, ignore_errors=True)
    rows = []
    for r, g, sp in zip(records, groups, split):
        name = f"{r['source']}__{r['sha256'][:16]}{r['image'].suffix.lower()}"
        img_out = OUT_DIR / "images" / sp / name
        lbl_out = OUT_DIR / "labels" / sp / f"{Path(name).stem}.txt"
        img_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["image"], img_out)
        lbl_out.write_text("\n".join(r["labels"]) + ("\n" if r["labels"] else ""), encoding="utf-8")
        rows.append({"image": img_out.relative_to(REPO_ROOT).as_posix(), "source": r["source"],
                     "sha256": r["sha256"], "group": int(g), "split": sp, "n_boxes": len(r["labels"])})
    with open(OUT_DIR / "split.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CONFIRMED_CLASSES)}\nnames: {json.dumps(CONFIRMED_CLASSES)}\n", encoding="utf-8")

    instances = defaultdict(Counter)
    images = defaultdict(Counter)
    for r, sp in zip(records, split):
        images[sp][r["target"]] += 1
        for line in r["labels"]:
            instances[sp][CONFIRMED_CLASSES[int(line.split()[0])]] += 1
    groups_by_split = defaultdict(set)
    for g, sp in zip(groups, split):
        groups_by_split[sp].add(int(g))
    evidence = {
        "classes": CONFIRMED_CLASSES,
        "gated_classes_excluded_by_rule_3": UNRELIABLE_CLASSES,
        "sources": source_report,
        "excluded_sources": excluded_sources,
        "include_non_mango": include_non_mango,
        "flagged_proxy_classes": sorted({s["target"] for s in SOURCES if s["slug"] not in excluded_sources
                                         and review.get(s["slug"], {}).get("host") != "mango"}),
        "n_images": len(records),
        "n_groups": int(groups.max() + 1),
        "images_per_split_class": {s: dict(images[s]) for s in ("train", "val", "test")},
        "instances_per_split_class": {s: dict(instances[s]) for s in ("train", "val", "test")},
        "images_without_boxes": sum(1 for r in records if not r["labels"]),
        "group_overlap_between_splits": {
            f"{a}&{b}": len(groups_by_split[a] & groups_by_split[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "seed": SEED,
        "env": collect_env(),
    }
    write_json(EVIDENCE, evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "env"}, indent=2, default=str))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download", "review", "build", "all"])
    parser.add_argument("--include-non-mango", action="store_true",
                        help="Include sources whose reviewed host is not mango, as FLAGGED proxy data (recorded)")
    a = parser.parse_args()
    if a.step in ("download", "all"):
        download(SOURCES)
    if a.step in ("review", "all"):
        review()
    if a.step in ("build", "all"):
        build(a.include_non_mango)
