"""
Shared pytest configuration.

Makes `ml.*` and `app.*` importable and points the backend at a throwaway database
and media directory so tests never write into the repository.
"""

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
for p in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="mango_tests_"))
os.environ.setdefault("MEDIA_DIR", str(_TEST_ROOT / "media"))
os.environ.setdefault("DATABASE_PATH", str(_TEST_ROOT / "test.db"))
os.environ.setdefault("JWT_SECRET", "test-secret-key-with-sufficient-length-0123456789")
# Never use real model weights in tests; tests opt in to a generated checkpoint explicitly.
os.environ.setdefault("DISEASE_WEIGHTS_PATH", str(_TEST_ROOT / "missing_disease.pth"))
os.environ.setdefault("PEST_WEIGHTS_PATH", str(_TEST_ROOT / "missing_pest.pt"))


import pytest  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402

from ml.disease_detection.inference import DISEASE_CLASSES  # noqa: E402


@pytest.fixture(scope="session")
def random_disease_checkpoint(tmp_path_factory):
    """A real EfficientNet-B3 checkpoint with random weights, in the format written by train.py."""
    torch.manual_seed(0)
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(DISEASE_CLASSES))
    path = tmp_path_factory.mktemp("weights") / "disease_classifier.pth"
    torch.save({"state_dict": model.state_dict(), "classes": list(DISEASE_CLASSES)}, path)
    return str(path)


class _Tensor1:
    """Minimal stand-in for a 1-element tensor row as returned by Ultralytics boxes."""
    def __init__(self, values):
        self._values = values

    def __getitem__(self, i):
        return self

    def item(self):
        return self._values[0]

    def tolist(self):
        return list(self._values)


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [_Tensor1([cls_id])]
        self.conf = [_Tensor1([conf])]
        self.xyxy = [_Tensor1(xyxy)]


class _Result:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYOLO:
    """Mimics the subset of the Ultralytics YOLO call interface used by PestDetector."""
    NAMES = {0: "Fruit Fly", 1: "Mealy Bug", 2: "Scale Insect", 3: "Mango Hopper"}

    def __init__(self, detections):
        self.detections = detections  # list of (cls_id, conf, [x1, y1, x2, y2])
        self.calls = []

    def __call__(self, image, conf=0.25, verbose=False):
        self.calls.append(conf)
        boxes = [_Box(c, p, b) for c, p, b in self.detections if p >= conf]
        return [_Result(boxes, self.NAMES)]


@pytest.fixture
def fake_yolo():
    return FakeYOLO
