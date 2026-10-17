"""
Pest detection unit tests: YOLOv8 output parsing and Inviolable Rule 3 (unreliable class flagging).
The YOLO model is replaced by a stub with the same call interface; no weights are needed.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from ml.pest_detection.inference import (
    CONFIRMED_CLASSES, UNRELIABLE_CLASSES, PestDetector, build_detection, load_host_verified,
)

SAMPLE_IMG = Image.new("RGB", (640, 640), color=(100, 140, 80))


def test_class_definitions():
    assert set(CONFIRMED_CLASSES) == {"Fruit Fly", "Mealy Bug", "Scale Insect"}
    assert set(UNRELIABLE_CLASSES) == {"Mango Hopper", "Stem Borer"}
    assert not set(CONFIRMED_CLASSES) & set(UNRELIABLE_CLASSES)


def test_build_detection_flags_unreliable_classes():
    for name in UNRELIABLE_CLASSES:
        det = build_detection(name, 0.9, [1, 2, 3, 4])
        assert det["is_unreliable"] is True
        assert "Sample size < 100" in det["warning"]
    det = build_detection("Fruit Fly", 0.9, [1, 2, 3, 4], host_verified=True)
    assert det["is_unreliable"] is False and det["warning"] is None


def test_missing_weights_returns_no_detections(tmp_path):
    detector = PestDetector(weights_path=str(tmp_path / "missing.pt"))
    assert detector.is_trained is False
    assert detector.predict(SAMPLE_IMG) == []


def test_filename_does_not_create_detections(tmp_path):
    """Regression: detections used to be fabricated from the image filename."""
    img_path = tmp_path / "mango_hopper_sample.jpg"
    SAMPLE_IMG.save(img_path)
    detector = PestDetector(weights_path=str(tmp_path / "missing.pt"))
    assert detector.predict(img_path) == []


def test_predict_parses_yolo_output_and_applies_threshold(tmp_path, fake_yolo):
    detector = PestDetector(weights_path=str(tmp_path / "missing.pt"))
    detector.model = fake_yolo([(0, 0.95, [10, 20, 30, 40]), (3, 0.5, [50, 50, 90, 90]), (2, 0.1, [0, 0, 5, 5])])

    dets = detector.predict(SAMPLE_IMG, conf_threshold=0.25)
    assert [d["pest_name"] for d in dets] == ["Fruit Fly", "Mango Hopper"]
    assert dets[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert dets[0]["is_unreliable"] is False
    assert dets[1]["is_unreliable"] is True
    assert detector.model.calls == [0.25]


def test_draw_detections_overlay(tmp_path):
    detector = PestDetector(weights_path=str(tmp_path / "missing.pt"))
    detections = [
        build_detection("Fruit Fly", 0.92, [100.0, 100.0, 200.0, 200.0]),
        build_detection("Mango Hopper", 0.85, [300.0, 300.0, 420.0, 420.0]),
    ]
    out_path = tmp_path / "overlay.jpg"
    annotated = detector.draw_detections(SAMPLE_IMG, detections, output_path=out_path)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (640, 640, 3)
    assert Path(out_path).is_file()
    # Boxes were actually drawn: the annotated image differs from the input
    assert not np.array_equal(annotated, np.array(SAMPLE_IMG))


def test_host_verified_defaults_to_false():
    det = build_detection("Fruit Fly", 0.9, [1, 2, 3, 4])
    assert det["host_verified"] is False and "not mango" in det["warning"]
    det = build_detection("Fruit Fly", 0.9, [1, 2, 3, 4], host_verified=True)
    assert det["host_verified"] is True and det["warning"] is None


def test_host_verified_is_read_from_deployment_metadata(tmp_path, fake_yolo):
    import json
    weights = tmp_path / "pest_detector.pt"
    (tmp_path / "pest_detector.json").write_text(json.dumps(
        {"host_verified": {"Fruit Fly": False, "Mealy Bug": True, "Scale Insect": False}}))
    assert load_host_verified(weights) == {"Fruit Fly": False, "Mealy Bug": True, "Scale Insect": False}
    assert load_host_verified(tmp_path / "other.pt") == {}
    detector = PestDetector(weights_path=str(weights))
    assert detector.has_host_verified_classes is True
    detector.model = fake_yolo([(0, 0.9, [0, 0, 9, 9]), (1, 0.9, [0, 0, 9, 9])])
    assert [d["host_verified"] for d in detector.predict(SAMPLE_IMG)] == [False, True]
    detector.host_verified = {"Fruit Fly": False}
    assert detector.has_host_verified_classes is False
