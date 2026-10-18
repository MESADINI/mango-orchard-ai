#!/usr/bin/env python3
"""Aggregates the per-dataset evidence files in results/data/ into results/data_stats.json."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.utils.experiment import git_info, write_json  # noqa: E402

DATA_DIR = REPO_ROOT / "results" / "data"
PARTS = {
    "mangoleafbd_download": "mangoleafbd_download.json",
    "disease": "disease_dataset.json",
    "external_test": "external_testset.json",
    "pest": "pest_dataset.json",
    "disease_dedup_calibration": "disease_dedup_calibration.json",
    "pest_source_review": "pest_review/summary.json",
}


def main() -> dict:
    stats = {"git": git_info(), "missing": []}
    for key, name in PARTS.items():
        path = DATA_DIR / name
        if not path.exists():
            stats["missing"].append(name)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("env", None)
        stats[key] = data
    out = write_json(REPO_ROOT / "results" / "data_stats.json", stats)
    print(f"Wrote {out}; missing parts: {stats['missing'] or 'none'}")
    return stats


if __name__ == "__main__":
    main()
