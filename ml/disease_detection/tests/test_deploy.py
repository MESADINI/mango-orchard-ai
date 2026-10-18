import json

import pytest
import timm
import torch

from ml.disease_detection import deploy as dep
from ml.disease_detection.inference import DISEASE_CLASSES, DiseaseDetector
from ml.disease_detection.train import sha256_file


def _setup(tmp_path, monkeypatch, tamper=False):
    results = tmp_path / "results"
    run = results / "final" / "mobilenetv3_large"
    run.mkdir(parents=True)
    m = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=len(DISEASE_CLASSES))
    torch.save({"state_dict": m.state_dict(), "classes": list(DISEASE_CLASSES), "arch": "mobilenetv3_large",
                "timm_id": "mobilenetv3_large_100", "img_size": 224}, run / "best.pth")
    sha = sha256_file(run / "best.pth")
    if tamper:
        with open(run / "best.pth", "ab") as f:
            f.write(b"x")
    (results / "summary.json").write_text(json.dumps({
        "selected": "mobilenetv3_large", "selection_rule": "highest mean CV macro-F1",
        "final": {"mobilenetv3_large": {"temperature": 1.7, "ood_threshold": 0.6,
                                        "summary": {"checkpoint_sha256": sha}}}}))
    monkeypatch.setattr(dep, "RESULTS", results)
    monkeypatch.setattr(dep, "REPO_ROOT", tmp_path)
    return tmp_path / "weights" / "disease_classifier.pth"


def test_deploy_embeds_temperature_threshold_and_provenance(tmp_path, monkeypatch):
    dest = _setup(tmp_path, monkeypatch)
    rec = dep.deploy(dest=dest)
    assert rec["sha256"] == sha256_file(dest)
    assert (tmp_path / "weights" / "disease_classifier.pth.sha256").read_text().startswith(rec["sha256"])
    d = DiseaseDetector(weights_path=str(dest), device="cpu")
    assert d.temperature == 1.7 and d.ood_threshold == 0.6 and d.arch == "mobilenetv3_large_100"


def test_deploy_refuses_checkpoint_that_changed_after_training(tmp_path, monkeypatch):
    dest = _setup(tmp_path, monkeypatch, tamper=True)
    with pytest.raises(RuntimeError):
        dep.deploy(dest=dest)
