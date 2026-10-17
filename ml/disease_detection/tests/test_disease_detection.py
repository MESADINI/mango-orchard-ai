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


def test_checkpoint_records_arch_size_and_temperature(tmp_path):
    """A new-format checkpoint of another backbone is rebuilt with its own arch, input size and temperature."""
    import timm
    model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=len(DISEASE_CLASSES))
    path = tmp_path / "m.pth"
    torch.save({"state_dict": model.state_dict(), "classes": list(DISEASE_CLASSES), "arch": "mobilenetv3_large",
                "timm_id": "mobilenetv3_large_100", "img_size": 224, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
                "temperature": 2.0, "ood_threshold": 1.01}, path)
    detector = DiseaseDetector(weights_path=str(path), device="cpu")
    assert detector.arch == "mobilenetv3_large_100" and detector.temperature == 2.0
    pred = detector.predict(Image.open(SAMPLE_LEAF))
    assert pred["disease_name"] in DISEASE_CLASSES
    assert abs(sum(pred["all_probabilities"].values()) - 1.0) < 1e-5
    assert pred["low_confidence"] is True          # every confidence is below a threshold of 1.01


def test_lr_schedule_warms_up_then_decays():
    from ml.disease_detection.train import DEFAULT_CONFIG, lr_at
    cfg = dict(DEFAULT_CONFIG, lr=1.0, warmup_epochs=2, epochs=10)
    assert lr_at(0, cfg) == 0.0 and lr_at(1, cfg) == 0.5 and lr_at(2, cfg) == 1.0
    assert lr_at(10, cfg) == pytest.approx(0.01) and lr_at(6, cfg) < lr_at(3, cfg)


def test_class_weights_inverse_frequency_mean_one():
    from ml.disease_detection.train import class_weights
    w = class_weights([0, 0, 0, 1], n_classes=3).tolist()
    assert w[2] == 0.0 and w[1] == pytest.approx(3 * w[0]) and (w[0] + w[1]) / 2 == pytest.approx(1.0)


def test_run_training_end_to_end_tiny(tmp_path):
    """Two epochs of a tiny model on synthetic images: writes checkpoint, sha256, history, logits."""
    from ml.disease_detection.train import run_training
    rows = []
    for i in range(20):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 48), ((i * 40) % 255, 120, 30)).save(p)
        rows.append({"abs_path": str(p), "path": p.name, "y": i % len(DISEASE_CLASSES), "group": str(i)})
    out = tmp_path / "run"
    s = run_training("mobilenetv3_small_050", rows[:15], rows[15:], out, {"test": rows[15:]},
                     {"epochs": 2, "warmup_epochs": 1, "batch_size": 8, "pretrained": False, "img_size": 64,
                      "patience": 5}, device="cpu")
    import hashlib, numpy as np
    assert (out / "best.pth").exists() and (out / "history.csv").exists() and (out / "config.json").exists()
    assert s["checkpoint_sha256"] == hashlib.sha256((out / "best.pth").read_bytes()).hexdigest()
    d = np.load(out / "logits_test.npz")
    assert d["logits"].shape == (5, len(DISEASE_CLASSES)) and list(d["y"]) == [r["y"] for r in rows[15:]]
    assert DiseaseDetector(weights_path=str(out / "best.pth"), device="cpu").is_trained
