#!/usr/bin/env python3
"""
Downloads MangoLeafBD reproducibly and verifies it against the authors' official release.

Source of images + labels:
    Hugging Face mirror Project-AgML/MangoLeafBD_disease_classification, pinned to a git revision,
    a single parquet file whose SHA-256 is checked after download. No credentials needed.
Verification:
    The authors' Mendeley Data record (doi:10.17632/hxsnvwty3r.1) publishes a SHA-256 for each of its
    4,000 files. Every extracted image is hashed and matched against that list, which proves the bytes
    are the original release and recovers the Mendeley folder -> class mapping independently.

Outputs (git-ignored):
    ml/datasets/raw/mangoleafbd/images/<Class>/<sha256[:16]>.jpg
    ml/datasets/raw/mangoleafbd/manifest.csv  (path, source_class, sha256, width, height, mendeley_folder)
Committed evidence:
    results/data/mangoleafbd_download.json
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.utils.experiment import collect_env, write_json  # noqa: E402

HF_REPO = "Project-AgML/MangoLeafBD_disease_classification"
HF_REVISION = "8efaea9716d692bf8b50ea1d0974bb3b098c3ff6"
HF_FILE = "data/train-00000-of-00001.parquet"
HF_SHA256 = "a9e8a5613de1f17be3e1c610215249dbbc23ee4519c94165b166e4250820b48c"
MENDELEY_API = "https://data.mendeley.com/public-api/datasets/hxsnvwty3r"
LICENSE = "CC BY-NC 3.0"
CITATION = ("Ahmed, S.I. et al. (2023). MangoLeafBD: A comprehensive image dataset to classify diseased and "
            "healthy mango leaves. Data in Brief 47, 108941. Data: doi:10.17632/hxsnvwty3r.1")

# Label index order from the HF dataset card (dataset_info.features.label.class_label.names).
LABEL_NAMES = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
               "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"]

RAW_DIR = REPO_ROOT / "ml" / "datasets" / "raw" / "mangoleafbd"
EVIDENCE = REPO_ROOT / "results" / "data" / "mangoleafbd_download.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, timeout: float = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mango-orchard-ai/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_parquet(dest: Path) -> Path:
    if dest.exists() and sha256_file(dest) == HF_SHA256:
        print(f"Parquet already present and verified: {dest}")
        return dest
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILE}"
    print(f"Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mango-orchard-ai/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    digest = sha256_file(tmp)
    if digest != HF_SHA256:
        tmp.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {HF_FILE}: got {digest}, expected {HF_SHA256}")
    tmp.replace(dest)
    return dest


def mendeley_files() -> list:
    """Returns the official Mendeley file list as [(sha256, folder_id), ...]."""
    record = json.loads(fetch(MENDELEY_API))
    return [(f["content_details"]["sha256_hash"], f["folder_id"]) for f in record["files"]]


def extract(parquet_path: Path, out_dir: Path) -> list:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet_path)
    rows = []
    for record in table.to_pylist():
        data = record["image"]["bytes"]
        cls = LABEL_NAMES[record["label"]]
        digest = sha256_bytes(data)
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            fmt = img.format
        ext = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"
        path = out_dir / "images" / cls / f"{digest[:16]}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        rows.append({"path": path.relative_to(REPO_ROOT).as_posix(), "source_class": cls,
                     "sha256": digest, "width": width, "height": height})
    return rows


def main(skip_mendeley: bool = False):
    parquet = download_parquet(RAW_DIR / "source.parquet")
    rows = extract(parquet, RAW_DIR)

    official = [] if skip_mendeley else mendeley_files()
    folder_by_hash = dict(official)
    for r in rows:
        r["mendeley_folder"] = folder_by_hash.get(r["sha256"], "")

    with open(RAW_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # A folder is consistent if every matched image in it carries the same HF label.
    folder_classes = defaultdict(Counter)
    for r in rows:
        if r["mendeley_folder"]:
            folder_classes[r["mendeley_folder"]][r["source_class"]] += 1
    folder_map = {fid: dict(c) for fid, c in folder_classes.items()}

    evidence = {
        "source": {"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "hf_file": HF_FILE,
                   "hf_sha256_verified": HF_SHA256, "mendeley_record": MENDELEY_API,
                   "license": LICENSE, "citation": CITATION},
        "n_images": len(rows),
        "n_unique_sha256": len({r["sha256"] for r in rows}),
        "per_class": dict(sorted(Counter(r["source_class"] for r in rows).items())),
        "image_sizes_top5": dict(Counter(f"{r['width']}x{r['height']}" for r in rows).most_common(5)),
        "n_distinct_image_sizes": len({(r["width"], r["height"]) for r in rows}),
        "mendeley_files": len(official),
        "mendeley_unique_sha256": len(folder_by_hash),
        "matched_to_mendeley_sha256": sum(1 for r in rows if r["mendeley_folder"]),
        "mendeley_folder_to_class": folder_map,
        "every_mendeley_folder_maps_to_one_class": all(len(c) == 1 for c in folder_map.values()),
        "env": collect_env(),
    }
    write_json(EVIDENCE, evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "env"}, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-mendeley", action="store_true", help="Skip the official SHA-256 cross-check")
    main(parser.parse_args().skip_mendeley)
