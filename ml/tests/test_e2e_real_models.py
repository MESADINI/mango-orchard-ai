"""
End-to-end checks with the real deployed weights (Phase 6). Skipped when the weights are not installed (they are
built by the training scripts and not committed). CPU only, so the results do not depend on a GPU.
"""

import csv
import json
from pathlib import Path

import pytest

from testing_support import REPO_ROOT, fake_weather

DISEASE_WEIGHTS = REPO_ROOT / "ml" / "disease_detection" / "weights" / "disease_classifier.pth"
PEST_WEIGHTS = REPO_ROOT / "ml" / "pest_detection" / "weights" / "pest_detector.pt"
SPLIT = REPO_ROOT / "ml" / "datasets" / "processed" / "disease" / "split.csv"

needs_disease = pytest.mark.skipif(not DISEASE_WEIGHTS.exists(), reason="deployed disease weights not installed")
needs_pest = pytest.mark.skipif(not PEST_WEIGHTS.exists(), reason="deployed pest weights not installed")


@pytest.fixture(scope="module")
def real_pipeline(tmp_path_factory):
    from ml.pipeline import MangoOrchardPipeline
    return MangoOrchardPipeline(disease_weights=str(DISEASE_WEIGHTS), pest_weights=str(PEST_WEIGHTS),
                                weather_fetcher=fake_weather, artifacts_dir=tmp_path_factory.mktemp("e2e"),
                                device="cpu")


@needs_disease
def test_installed_weights_match_the_registry():
    from ml.model_registry import status
    assert status()["disease_classifier"]["status"] == "ok"


@needs_disease
def test_deployed_checkpoint_carries_its_calibration():
    from ml.disease_detection.inference import DiseaseDetector
    det = DiseaseDetector(str(DISEASE_WEIGHTS), device="cpu")
    reg = json.loads((REPO_ROOT / "ml" / "model_registry.json").read_text(encoding="utf-8"))
    entry = next(m for m in reg["models"] if m["name"] == "disease_classifier")
    assert det.classes == entry["classes"]
    assert det.temperature == pytest.approx(entry["calibration"]["temperature"])
    assert det.ood_threshold == pytest.approx(entry["calibration"]["low_confidence_threshold"])


@needs_disease
@pytest.mark.skipif(not SPLIT.exists(), reason="disease split not built")
def test_real_model_on_test_split_images(real_pipeline):
    """One test-split image per class: the deployed model scored 1.000 on this split (results/disease)."""
    with open(SPLIT, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    for label in sorted({r["label"] for r in rows}):
        r = next(r for r in rows if r["label"] == label)
        out = real_pipeline.analyze_orchard(REPO_ROOT / r["path"], scan_type="disease", generate_gradcam=False)
        d = out["disease_diagnosis"]
        assert d["model_available"] is True
        if d["low_confidence"]:                       # the retake path, never a silent guess
            assert "disease" in out["orchard_health_index"]["missing_components"]
        else:
            assert d["disease_name"] == label


@needs_disease
def test_real_pipeline_full_response(real_pipeline):
    leaf = REPO_ROOT / "ml" / "datasets" / "sample_leaves" / "anthracnose_sample.jpg"
    out = real_pipeline.analyze_orchard(leaf, generate_gradcam=True)
    assert out["status"] == "success"
    assert Path(out["disease_diagnosis"]["gradcam_overlay_path"]).is_file()
    ohi = out["orchard_health_index"]
    assert ohi["nutrient_input_was_defaulted"] is True
    assert ohi["ohi_score"] is None or 0 <= ohi["ohi_score"] <= 100


@needs_pest
def test_real_pest_detector_is_flagged_host_unverified(real_pipeline):
    pest = REPO_ROOT / "ml" / "datasets" / "sample_pests" / "fruit_fly_sample.jpg"
    out = real_pipeline.analyze_orchard(None, pest, scan_type="pest")
    p = out["pest_intelligence"]
    assert p["model_available"] is True and p["model_host_verified"] is False
    assert all(d["host_verified"] is False for d in p["detections"])
    assert "pest" in out["orchard_health_index"]["missing_components"]
