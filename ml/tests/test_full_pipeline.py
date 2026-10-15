"""
End-to-end pipeline integration tests.

Covers: honest behaviour without trained weights, the trained-model path (random-weight
checkpoint), Grad-CAM output, scan_type handling, Rule 2 citations, Rule 3 flagging,
Rule 4 nutrient disclosure and offline-weather disclosure.
"""

from pathlib import Path

import pytest

from testing_support import SAMPLE_LEAF, SAMPLE_PEST, fake_weather, offline_weather
from ml.disease_detection.inference import DISEASE_CLASSES
from ml.pipeline import MangoOrchardPipeline


@pytest.fixture
def untrained_pipeline(tmp_path):
    return MangoOrchardPipeline(
        disease_weights=str(tmp_path / "none.pth"),
        pest_weights=str(tmp_path / "none.pt"),
        weather_fetcher=fake_weather,
        artifacts_dir=tmp_path,
    )


@pytest.fixture
def trained_pipeline(tmp_path, random_disease_checkpoint, fake_yolo):
    pipeline = MangoOrchardPipeline(
        disease_weights=random_disease_checkpoint,
        pest_weights=str(tmp_path / "none.pt"),
        weather_fetcher=fake_weather,
        artifacts_dir=tmp_path,
    )
    pipeline.pest_detector.model = fake_yolo([
        (0, 0.91, [10, 10, 60, 60]),      # Fruit Fly (confirmed)
        (3, 0.80, [100, 100, 150, 150]),  # Mango Hopper (Rule 3: unreliable)
        (1, 0.10, [5, 5, 20, 20]),        # below the confidence threshold
    ])
    return pipeline


def test_untrained_pipeline_reports_unavailable_instead_of_guessing(untrained_pipeline):
    result = untrained_pipeline.analyze_orchard(SAMPLE_LEAF, SAMPLE_PEST)

    assert result["model_status"] == {"disease_model_trained": False, "pest_model_trained": False}
    d = result["disease_diagnosis"]
    assert d["model_available"] is False
    assert d["disease_name"] == "Unavailable"
    assert d["confidence"] == 0.0
    assert d["gradcam_overlay_path"] is None
    assert result["pest_intelligence"]["detections"] == []

    ohi = result["orchard_health_index"]
    assert ohi["is_partial"] is True
    assert set(ohi["missing_components"]) == {"disease", "pest"}
    assert ohi["penalty_breakdown"]["disease_severity_penalty"] == 0.0
    assert any("Disease classification unavailable" in c for c in ohi["caveats"])
    assert any("Pest detection unavailable" in c for c in ohi["caveats"])
    # No treatment advice without a diagnosis
    assert result["treatment_protocols"]["protocols"] == []


def test_trained_pipeline_schema_gradcam_and_rules(trained_pipeline, tmp_path):
    result = trained_pipeline.analyze_orchard(
        SAMPLE_LEAF, SAMPLE_PEST, output_dir=tmp_path, output_prefix="scan1"
    )

    d = result["disease_diagnosis"]
    assert d["model_available"] is True
    assert d["disease_name"] in DISEASE_CLASSES
    assert 0.0 <= d["confidence"] <= 1.0
    assert set(d["probabilities"]) == set(DISEASE_CLASSES)
    assert abs(sum(d["probabilities"].values()) - 1.0) < 1e-4
    assert d["leaf_pixels"] > 0
    assert d["gradcam_overlay_path"] == str(tmp_path / "scan1_gradcam.jpg")
    assert Path(d["gradcam_overlay_path"]).is_file()

    p = result["pest_intelligence"]
    names = sorted(det["pest_name"] for det in p["detections"])
    assert names == ["Fruit Fly", "Mango Hopper"]
    assert p["confirmed_count"] == 1 and p["unreliable_count"] == 1
    assert p["has_unreliable_classes"] is True
    assert "sample size < 100" in p["unreliable_warning"].lower()
    assert Path(p["pest_overlay_path"]).is_file()

    ohi = result["orchard_health_index"]
    assert 0.0 <= ohi["ohi_score"] <= 100.0
    assert ohi["nutrient_input_was_defaulted"] is True  # Rule 4
    assert ohi["is_partial"] is False

    # Rule 2: every protocol is cited; Rule 3: no protocol is issued for the unverified pest
    protocols = result["treatment_protocols"]["protocols"]
    conditions = {r["condition"] for r in protocols}
    assert "Mango Hopper" not in conditions
    assert "Fruit Fly" in conditions
    for rec in protocols:
        assert any(src in rec["citation_source"] for src in ["ICAR", "TNAU", "DPPQS", "Agritech"]), rec


def test_scan_type_disease_skips_pests(trained_pipeline):
    result = trained_pipeline.analyze_orchard(SAMPLE_LEAF, SAMPLE_PEST, scan_type="disease")
    assert result["pest_intelligence"]["performed"] is False
    assert result["pest_intelligence"]["detections"] == []
    assert trained_pipeline.pest_detector.model.calls == []
    assert "pest" in result["orchard_health_index"]["missing_components"]


def test_scan_type_pest_skips_disease(trained_pipeline):
    result = trained_pipeline.analyze_orchard(None, SAMPLE_PEST, scan_type="pest")
    assert result["disease_diagnosis"]["performed"] is False
    assert result["disease_diagnosis"]["disease_name"] is None
    assert result["pest_intelligence"]["total_detections"] == 2


def test_invalid_scan_type_rejected(untrained_pipeline):
    with pytest.raises(ValueError):
        untrained_pipeline.analyze_orchard(SAMPLE_LEAF, scan_type="everything")


def test_offline_weather_is_disclosed(tmp_path):
    pipeline = MangoOrchardPipeline(
        disease_weights=str(tmp_path / "none.pth"),
        pest_weights=str(tmp_path / "none.pt"),
        weather_fetcher=offline_weather,
    )
    result = pipeline.analyze_orchard(SAMPLE_LEAF, scan_type="disease")
    assert result["weather_vulnerability"]["live_data"] is False
    assert result["weather_vulnerability"]["7_day_forecast"] == []
    assert "weather" in result["orchard_health_index"]["missing_components"]
    assert any("Live weather data unavailable" in c for c in result["orchard_health_index"]["caveats"])


def test_weather_forecast_handles_null_values(untrained_pipeline):
    result = untrained_pipeline.analyze_orchard(SAMPLE_LEAF, scan_type="disease")
    forecast = result["weather_vulnerability"]["7_day_forecast"]
    assert len(forecast) == 7
    assert forecast[6]["temperature_max"] == 30.0  # null replaced by documented default
    assert result["weather_vulnerability"]["current_relative_humidity_pct"] == 85.0
