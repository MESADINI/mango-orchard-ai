"""Model registry: entries built from the deployment records, and installed-weights verification by SHA-256."""

import hashlib
import json

from ml import model_registry as R


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def test_build_and_status(tmp_path):
    weights = tmp_path / "ml" / "disease_detection" / "weights" / "disease_classifier.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    deployed = _write(tmp_path / "d.json", {
        "arch": "efficientnet_b3", "sha256": sha, "deployed_to": "ml/disease_detection/weights/disease_classifier.pth",
        "temperature": 1.0, "ood_threshold": 0.8, "source_run": "r", "source_sha256": "s", "git": {"commit": "c"}})
    ev = _write(tmp_path / "e.json", {"test": {"accuracy": 1.0, "macro_f1": 1.0, "n": 3, "confusion_matrix": []}})
    pest = _write(tmp_path / "p.json", {"model": "yolo11n", "sha256": "f" * 64, "classes": ["Fruit Fly"],
                                        "host_verified": {"Fruit Fly": False}, "source_checkpoint": "x"})
    reg = R.build(deployed, ev, pest, out=tmp_path / "registry.json")
    names = [m["name"] for m in reg["models"]]
    assert names == ["disease_classifier", "pest_detector"]
    disease = reg["models"][0]
    assert disease["version"] == sha[:12] and disease["calibration"]["low_confidence_threshold"] == 0.8
    assert disease["metrics"]["test"] == {"accuracy": 1.0, "macro_f1": 1.0, "low_confidence_fraction": None, "n": 3}

    st = R.status(tmp_path / "registry.json", root=tmp_path)
    assert st["disease_classifier"]["status"] == "ok"
    assert st["pest_detector"]["status"] == "missing" and st["pest_detector"]["host_verified"] == {"Fruit Fly": False}
    weights.write_bytes(b"other")
    assert R.status(tmp_path / "registry.json", root=tmp_path)["disease_classifier"]["status"] == "sha_mismatch"


def test_missing_records_give_an_empty_registry(tmp_path):
    reg = R.build(tmp_path / "a", tmp_path / "b", tmp_path / "c", out=tmp_path / "r.json")
    assert reg["models"] == [] and R.status(tmp_path / "r.json") == {}


def test_committed_registry_is_consistent():
    reg = json.loads(R.REGISTRY.read_text(encoding="utf-8"))
    for m in reg["models"]:
        assert m["version"] == m["sha256"][:12] and len(m["sha256"]) == 64 and m["classes"]
