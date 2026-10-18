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
    # as if every confirmed class had been trained on mango images (the deployed model is not: see below)
    pipeline.pest_detector.host_verified = {"Fruit Fly": True, "Mealy Bug": True, "Scale Insect": True}
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
    # not assessed is not healthy: no penalty value, and without any image component there is no OHI at all
    assert ohi["penalty_breakdown"]["disease_severity_penalty"] is None
    assert ohi["ohi_score"] is None and ohi["health_status"].startswith("Unknown")
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
    assert result["weather_vulnerability"]["current_temperature_c"] is None      # no invented defaults
    assert result["orchard_health_index"]["penalty_breakdown"]["weather_vulnerability_index"] is None


def test_offline_weather_is_excluded_not_defaulted(tmp_path, random_disease_checkpoint):
    pipeline = MangoOrchardPipeline(
        disease_weights=random_disease_checkpoint,
        pest_weights=str(tmp_path / "none.pt"),
        weather_fetcher=offline_weather,
        artifacts_dir=tmp_path,
    )
    pipeline.disease_detector.ood_threshold = None
    ohi = pipeline.analyze_orchard(SAMPLE_LEAF, scan_type="disease", generate_gradcam=False)["orchard_health_index"]
    assert ohi["ohi_score"] is not None
    assert ohi["weights_used"]["weather"] == 0.0 and ohi["weights_used"]["disease"] == 1.0


def test_low_confidence_prediction_is_excluded_and_asks_for_retake(trained_pipeline):
    trained_pipeline.disease_detector.ood_threshold = 1.01      # every prediction is below the threshold
    result = trained_pipeline.analyze_orchard(SAMPLE_LEAF, SAMPLE_PEST, generate_gradcam=False)
    assert result["disease_diagnosis"]["low_confidence"] is True
    ohi = result["orchard_health_index"]
    assert "disease" in ohi["missing_components"]
    assert ohi["penalty_breakdown"]["disease_severity_penalty"] is None
    assert any("retake the photo" in c for c in ohi["caveats"])
    conditions = {r["condition"] for r in result["treatment_protocols"]["protocols"]}
    assert result["disease_diagnosis"]["disease_name"] not in conditions   # no disease treatment when unsure


def test_unverified_pests_do_not_enter_the_ohi(trained_pipeline):
    ohi = trained_pipeline.analyze_orchard(SAMPLE_LEAF, SAMPLE_PEST, generate_gradcam=False)["orchard_health_index"]
    assert ohi["excluded_unverified_pest_detections"] == 1                 # the Mango Hopper box
    assert ohi["penalty_breakdown"]["pest_pressure_penalty"] == round(18.0 * 0.91, 2)   # Fruit Fly only


def test_pest_model_trained_off_host_is_excluded_from_the_ohi(trained_pipeline):
    """The deployed detector learned every class from traps / grape / coffee images (host_verified False)."""
    trained_pipeline.pest_detector.host_verified = {"Fruit Fly": False, "Mealy Bug": False, "Scale Insect": False}
    result = trained_pipeline.analyze_orchard(SAMPLE_LEAF, SAMPLE_PEST, generate_gradcam=False)
    p, ohi = result["pest_intelligence"], result["orchard_health_index"]
    fly = next(d for d in p["detections"] if d["pest_name"] == "Fruit Fly")
    assert fly["host_verified"] is False and "not mango" in fly["warning"]
    assert p["confirmed_count"] == 0 and p["host_unverified_count"] == 1 and p["model_host_verified"] is False
    assert "pest" in ohi["missing_components"] and ohi["penalty_breakdown"]["pest_pressure_penalty"] is None
    assert any("non-mango" in c for c in ohi["caveats"])
    assert "Fruit Fly" not in {r["condition"] for r in result["treatment_protocols"]["protocols"]}


def test_weather_forecast_handles_null_values(untrained_pipeline):
    result = untrained_pipeline.analyze_orchard(SAMPLE_LEAF, scan_type="disease")
    forecast = result["weather_vulnerability"]["7_day_forecast"]
    assert len(forecast) == 7
    assert forecast[6]["temperature_max"] is None          # null is kept, never replaced by a default
    assert forecast[6]["powdery_mildew_risk"] == "Unknown"
    assert result["weather_vulnerability"]["current_relative_humidity_pct"] == 85.0
