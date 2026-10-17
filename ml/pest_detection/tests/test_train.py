"""Phase 3 experiment driver: host review, model selection on val only, SHA-verified deterministic deployment."""

import json

import pytest

from ml.pest_detection import train as T


def test_host_verified_requires_a_mango_review(tmp_path):
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"fruit_fly": {"host": "other:trap"}, "mealy_bug": {"host": "mango"}}))
    assert T.host_verified_by_class(review) == {"Fruit Fly": False, "Mealy Bug": True, "Scale Insect": False}
    assert T.host_verified_by_class(tmp_path / "missing.json") == {c: False for c in T.CONFIRMED_CLASSES}


def test_repository_pest_sources_are_not_mango():
    assert not any(T.host_verified_by_class().values())


def _summary(val, test, params):
    per = {c: {"precision": 0.5, "recall": 0.5, "ap50": 0.5, "ap50_95": test} for c in T.CONFIRMED_CLASSES}
    m = {"precision": 0.5, "recall": 0.5, "map50": 0.5, "per_class": per}
    return {"val": {**m, "map50_95": val}, "test": {**m, "map50_95": test}, "n_params": params, "epochs_run": 3,
            "names": list(T.CONFIRMED_CLASSES)}


def test_selection_ignores_the_test_split():
    s = {"a": _summary(0.60, 0.99, 3e6), "b": _summary(0.70, 0.10, 9e6)}
    assert T.select_model(s) == "b"
    s["c"] = _summary(0.70, 0.10, 2e6)                       # tie on val -> fewer parameters
    assert T.select_model(s) == "c"


def test_deploy_copies_the_selected_checkpoint_and_verifies_it(tmp_path, monkeypatch):
    results, weights = tmp_path / "results", tmp_path / "weights"
    monkeypatch.setattr(T, "REPO_ROOT", tmp_path)
    for name, val, payload in (("yolov8n", 0.5, b"n"), ("yolo11n", 0.6, b"eleven")):
        ckpt = tmp_path / "runs" / name / "best.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(payload)
        s = _summary(val, 0.4, 3e6)
        s.update(checkpoint=f"runs/{name}/best.pt", checkpoint_sha256=T.sha256_file(ckpt))
        (results / name).mkdir(parents=True)
        (results / name / "summary.json").write_text(json.dumps(s))
    meta = T.deploy(results, weights)
    assert meta["model"] == "yolo11n"
    assert (weights / "pest_detector.pt").read_bytes() == b"eleven"
    assert json.loads((weights / "pest_detector.json").read_text())["sha256"] == meta["sha256"]
    assert (results / "models.md").exists() and (results / "deployed.json").exists()

    (tmp_path / "runs" / "yolo11n" / "best.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        T.deploy(results, weights)
