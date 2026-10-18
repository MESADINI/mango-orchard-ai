"""
Integrity checks on the built datasets. Skipped when a dataset has not been built (e.g. a fresh clone);
run `python tasks.py data` first to exercise them.
"""

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from ml.datasets.build_pest_dataset import CONFIRMED_CLASSES, parse_label_line
from ml.disease_detection.inference import DISEASE_CLASSES

REPO = Path(__file__).resolve().parents[3]
DISEASE_SPLIT = REPO / "ml" / "datasets" / "processed" / "disease" / "split.csv"
PEST_SPLIT = REPO / "ml" / "datasets" / "processed" / "pest" / "split.csv"
EXTERNAL_MANIFEST = REPO / "ml" / "datasets" / "processed" / "external" / "manifest.csv"


def _rows(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} not built")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _no_overlap(rows, key):
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r[key])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (by_split[a] & by_split[b]), f"{key} overlap between {a} and {b}"


# ----------------------------------------------------------------------------- disease
def test_disease_splits_share_no_images_or_groups():
    rows = _rows(DISEASE_SPLIT)
    _no_overlap(rows, "sha256")
    _no_overlap(rows, "group")


def test_disease_labels_and_files_are_consistent():
    rows = _rows(DISEASE_SPLIT)
    assert {r["label"] for r in rows} == set(DISEASE_CLASSES)
    for r in rows:
        path = REPO / r["path"]
        assert path.exists(), r["path"]
        assert path.parent.name == r["label"] and path.parent.parent.name == r["split"]


def test_disease_cv_folds_exclude_test_and_respect_groups():
    rows = _rows(DISEASE_SPLIT)
    assert all(r["cv_fold"] == "-1" for r in rows if r["split"] == "test")
    pool = [r for r in rows if r["split"] != "test"]
    assert {r["cv_fold"] for r in pool} <= {"0", "1", "2", "3", "4", "-2"}
    assert {"0", "1", "2", "3", "4"} <= {r["cv_fold"] for r in pool}
    assert all(r["split"] == "train" for r in pool if r["cv_fold"] == "-2")
    fold_of_group = defaultdict(set)
    for r in pool:
        fold_of_group[r["group"]].add(r["cv_fold"])
    assert all(len(f) == 1 for f in fold_of_group.values()), "a duplicate group spans several CV folds"


def test_disease_split_proportions_are_roughly_70_15_15():
    # measured over freely split images; oversized groups (cv_fold -2) are train-only by design
    rows = [r for r in _rows(DISEASE_SPLIT) if r["cv_fold"] != "-2"]
    n = len(rows)
    for split, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        frac = sum(r["split"] == split for r in rows) / n
        assert abs(frac - target) < 0.03, (split, frac)


# ----------------------------------------------------------------------------- external test set
def test_external_set_has_no_overlap_with_training_data():
    ext = _rows(EXTERNAL_MANIFEST)
    train = {r["sha256"] for r in _rows(DISEASE_SPLIT)}
    assert not ({r["sha256"] for r in ext} & train)
    assert {r["label"] for r in ext} <= set(DISEASE_CLASSES)
    assert all(r["near_duplicate_of_training"] == "False" for r in ext)
    assert all(r["mangoleafbd_format"] == "False" for r in ext), "MangoLeafBD-format images must be excluded"


# ----------------------------------------------------------------------------- pest
def test_pest_splits_share_no_images_or_groups():
    rows = _rows(PEST_SPLIT)
    _no_overlap(rows, "sha256")
    _no_overlap(rows, "group")


def test_pest_labels_are_valid_and_never_gated():
    rows = _rows(PEST_SPLIT)
    for r in rows:
        image = REPO / r["image"]
        label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
        assert image.exists() and label.exists()
        for line in label.read_text(encoding="utf-8").splitlines():
            parsed = parse_label_line(line)
            assert parsed is not None, (label, line)
            assert parsed[0] < len(CONFIRMED_CLASSES), "label outside confirmed classes (Rule 3)"
