"""Disease detector and training-label tests."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from testing_support import SAMPLE_LEAF
from ml.disease_detection.inference import DISEASE_CLASSES, DiseaseDetector, load_checkpoint
from ml.disease_detection.train import load_split


def test_missing_weights_do_not_crash_and_report_unavailable(tmp_path):
    assert DiseaseDetector(weights_path=None).is_trained is False

    detector = DiseaseDetector(weights_path=str(tmp_path / "missing.pth"))
    pred = detector.predict(Image.open(SAMPLE_LEAF))
    assert pred["model_available"] is False
    assert pred["disease_name"] == "Unavailable"
    assert pred["all_probabilities"] == {}


def test_trained_checkpoint_predicts_valid_class(random_disease_checkpoint):
    detector = DiseaseDetector(weights_path=random_disease_checkpoint, device="cpu")
    assert detector.is_trained
    pred = detector.predict(Image.open(SAMPLE_LEAF))
    assert pred["disease_name"] in DISEASE_CLASSES
    assert pred["model_available"] is True


def test_legacy_raw_state_dict_uses_alphabetical_class_order(tmp_path):
    path = tmp_path / "legacy.pth"
    torch.save({"weight": torch.zeros(1)}, path)
    ckpt = load_checkpoint(str(path), map_location="cpu")
    assert ckpt["classes"] == sorted(DISEASE_CLASSES)


def test_training_labels_follow_canonical_class_order(tmp_path):
    """Regression: ImageFolder's alphabetical indices must be remapped to DISEASE_CLASSES order."""
    for name in ["Healthy", "Anthracnose", "Sooty Mold"]:
        (tmp_path / name).mkdir()
        Image.new("RGB", (8, 8), (0, 128, 0)).save(tmp_path / name / "a.jpg")
    (tmp_path / "Powdery Mildew").mkdir()  # empty class folder must be tolerated

    ds = load_split(str(tmp_path), transform=None)
    assert ds.classes == DISEASE_CLASSES
    labels = {Path(path).parent.name: target for path, target in ds.samples}
    assert labels == {name: DISEASE_CLASSES.index(name) for name in ["Healthy", "Anthracnose", "Sooty Mold"]}


def test_leaf_spot_is_not_a_model_class(tmp_path):
    """Leaf Spot has no training data (absent from MangoLeafBD); a folder for it must be rejected."""
    assert "Leaf Spot" not in DISEASE_CLASSES
    (tmp_path / "Leaf Spot").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "Leaf Spot" / "a.jpg")
    with pytest.raises(ValueError):
        load_split(str(tmp_path), transform=None)
