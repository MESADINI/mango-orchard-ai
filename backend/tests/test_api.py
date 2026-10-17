"""
Backend API test suite (FastAPI TestClient).

Runs against a temporary SQLite database and media directory (configured in the
repository-root conftest.py) with Open-Meteo stubbed out, so it needs no network.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from testing_support import SAMPLE_LEAF, SAMPLE_PEST, fake_weather, offline_weather
from app.core.config import settings
from app.main import app
from app.services.pipeline_service import pipeline_service
import app.api.weather as weather_api

DEMO = {"email": settings.DEMO_USER_EMAIL, "password": settings.DEMO_USER_PASSWORD}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan (seeds demo account, loads pipeline)
        pipeline_service.get_pipeline().weather_fetcher = fake_weather
        yield c


@pytest.fixture(scope="module")
def auth(client):
    res = client.post("/api/auth/login", json=DEMO)
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _analyze(client, auth, **data):
    with open(SAMPLE_LEAF, "rb") as lf, open(SAMPLE_PEST, "rb") as pf:
        return client.post(
            "/api/analyze/",
            files={"file": ("leaf.jpg", lf, "image/jpeg"), "pest_file": ("pest.jpg", pf, "image/jpeg")},
            data={"scan_type": "dual", **data},
            headers=auth,
        )


# ---------------- Health ----------------
def test_health_reports_rules_and_model_status(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert "Rule 3: Unreliable Class Flagging" in data["models"]["rules_enforced"]
    assert data["models"]["disease_model_trained"] is False
    assert data["models"]["pest_model_trained"] is False


# ---------------- Auth ----------------
def test_login_and_me(client, auth):
    me = client.get("/api/auth/me", headers=auth)
    assert me.status_code == 200
    assert me.json()["email"] == DEMO["email"]


def test_login_rejects_wrong_password_and_unknown_user(client):
    assert client.post("/api/auth/login", json={**DEMO, "password": "wrong"}).status_code == 401
    # Regression: any email used to be accepted with the demo password
    res = client.post("/api/auth/login", json={"email": "stranger@example.com", "password": DEMO["password"]})
    assert res.status_code == 401


def test_register_forces_grower_role_and_rejects_duplicates(client):
    res = client.post("/api/auth/register", json={
        "email": "new.grower@example.com", "password": "longpassword1", "full_name": "New Grower", "role": "admin",
    })
    assert res.status_code == 200, res.text
    assert res.json()["user"]["role"] == "grower"
    dup = client.post("/api/auth/register", json={"email": "NEW.grower@example.com", "password": "longpassword1"})
    assert dup.status_code == 400


def test_register_validates_input(client):
    assert client.post("/api/auth/register", json={"email": "x@y.com", "password": "short"}).status_code == 422
    assert client.post("/api/auth/register", json={"email": "not-an-email", "password": "longpassword1"}).status_code == 422


def test_protected_endpoints_require_token(client):
    assert client.get("/api/farm/").status_code == 401
    assert client.get("/api/analyze/").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


# ---------------- Analysis ----------------
def test_analyze_dual_scan_is_honest_without_weights(client, auth):
    res = _analyze(client, auth, latitude=12.5266, longitude=78.2147)
    assert res.status_code == 200, res.text
    data = res.json()

    # neither disease nor pest could be assessed: no index rather than a weather-only number
    assert data["ohi_score"] is None and data["health_status"].startswith("Unknown")
    assert data["penalty_breakdown"]["disease_severity_penalty"] is None
    assert data["nutrient_input_was_defaulted"] is True
    assert data["model_status"] == {"disease_model_trained": False, "pest_model_trained": False}
    assert data["disease_results"] == []  # no fabricated diagnosis
    assert data["pest_results"] == []
    assert data["ohi_is_partial"] is True
    assert any("Disease classification unavailable" in c for c in data["caveats"])
    assert data["lesion_severity_percentage"] is not None
    assert data["weather_vulnerability"]["live_data"] is True
    assert data["image_url"].startswith("/media/uploads/") and data["image_url"].endswith("_leaf.jpg")

    # The uploaded image is served back
    assert client.get(data["image_url"]).status_code == 200


def test_analyze_with_trained_disease_model(client, auth, monkeypatch, random_disease_checkpoint):
    from ml.disease_detection.inference import DISEASE_CLASSES, DiseaseDetector

    pipeline = pipeline_service.get_pipeline()
    monkeypatch.setattr(pipeline, "disease_detector", DiseaseDetector(random_disease_checkpoint, device="cpu"))

    res = _analyze(client, auth, scan_type="disease")
    assert res.status_code == 200, res.text
    data = res.json()
    assert len(data["disease_results"]) == 1
    d = data["disease_results"][0]
    assert d["disease_name"] in DISEASE_CLASSES
    assert d["gradcam_overlay_url"].startswith("/media/results/")
    assert client.get(d["gradcam_overlay_url"]).status_code == 200
    for rec in data["recommendations"]:
        assert any(src in rec["citation_source"] for src in ["ICAR", "TNAU", "DPPQS", "Agritech"])


def test_analyze_rejects_bad_inputs(client, auth):
    # Not an image, even though it claims to be one (and would previously be saved as .html)
    res = client.post("/api/analyze/", files={"file": ("x.html", b"<script>alert(1)</script>", "image/png")},
                      data={"scan_type": "disease"}, headers=auth)
    assert res.status_code == 400

    # Unsupported image format
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, "BMP")
    res = client.post("/api/analyze/", files={"file": ("x.bmp", buf.getvalue(), "image/bmp")},
                      data={"scan_type": "disease"}, headers=auth)
    assert res.status_code == 400

    # Invalid scan type and coordinates
    assert _analyze(client, auth, scan_type="everything").status_code == 422
    assert _analyze(client, auth, latitude=123).status_code == 422

    # Missing leaf image for a disease scan
    res = client.post("/api/analyze/", data={"scan_type": "disease"}, headers=auth)
    assert res.status_code == 400


def test_upload_size_limit(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)
    res = _analyze(client, auth)
    assert res.status_code == 400
    assert "limit" in res.json()["detail"]


def test_scan_history_and_retrieval_are_per_user(client, auth):
    created = _analyze(client, auth).json()
    history = client.get("/api/analyze/", headers=auth).json()
    assert history[0]["scan_id"] == created["scan_id"]
    assert client.get(f"/api/analyze/{created['scan_id']}", headers=auth).status_code == 200

    other = client.post("/api/auth/register", json={"email": "other@example.com", "password": "longpassword1"})
    other_auth = {"Authorization": f"Bearer {other.json()['access_token']}"}
    assert client.get(f"/api/analyze/{created['scan_id']}", headers=other_auth).status_code == 404
    assert client.get("/api/analyze/", headers=other_auth).json() == []


# ---------------- Reports ----------------
def test_report_reflects_real_scan_data(client, auth):
    scan = _analyze(client, auth).json()
    res = client.get(f"/api/report/generate/{scan['scan_id']}", headers=auth)
    assert res.status_code == 200
    assert res.headers["content-type"].split(";")[0] in ("application/pdf", "text/html")
    if res.headers["content-type"].startswith("text/html"):
        body = res.text
        assert scan["ohi_score"] is None and "not assessed" in body   # untrained models in the test setup
        assert scan["scan_id"] in body
        assert "82.5" not in body  # regression: the score used to be hard-coded


def test_report_renders_scores_and_unassessed_components():
    from app.api.report import render_report_html
    scan = {"scan_id": "abc", "created_at": "2026-10-17T00:00:00Z", "scan_type": "dual", "ohi_score": 71.25,
            "health_status": "Mild Stress / Warning", "badge_color": "#F59E0B", "disease_results": [],
            "pest_results": [], "recommendations": [], "caveats": [], "ohi_is_partial": True,
            "ohi_missing_components": ["weather"], "latitude": 12.5, "longitude": 78.2,
            "penalty_breakdown": {"disease_severity_penalty": 40.0, "pest_pressure_penalty": 10.0,
                                  "weather_vulnerability_index": None, "nutrient_penalty": None,
                                  "weighted_total_penalty": 28.75},
            "weights_used": {"disease": 0.5333, "pest": 0.4667, "weather": 0.0, "nutrient": 0.0},
            "weather_vulnerability": {"live_data": False, "current_temperature_c": None}}
    body = render_report_html(scan)
    assert "71.2 / 100" in body or "71.3 / 100" in body
    assert "not assessed" in body and "Live weather data unavailable" in body


def test_report_renders_recommendations_with_and_without_pre_harvest_interval():
    """Regression: a missing '+' made the treatment cell call a string, crashing every report with treatments."""
    from app.api.report import render_report_html
    rec = {"condition": "Anthracnose", "treatment_type": "Chemical", "product_name": "Carbendazim",
           "dosage_rate": "0.1%", "application_guidance": "Spray at flowering.", "citation_source": "NHM (2012)"}
    scan = {"scan_id": "r1", "created_at": "2026-10-17T00:00:00Z", "scan_type": "disease", "ohi_score": None,
            "health_status": "Unknown", "badge_color": "#6B7280", "disease_results": [], "pest_results": [],
            "recommendations": [{**rec, "waiting_period_days": 15}, {**rec, "waiting_period_days": None}],
            "caveats": [], "ohi_is_partial": True, "ohi_missing_components": [], "latitude": 12.5,
            "longitude": 78.2, "penalty_breakdown": {}, "weights_used": {}, "weather_vulnerability": {}}
    body = render_report_html(scan)
    assert "Last spray at least 15 days before harvest" in body
    assert "Check the product label for the pre-harvest interval" in body


def test_report_unknown_scan_is_404(client, auth):
    assert client.get("/api/report/generate/does-not-exist", headers=auth).status_code == 404


# ---------------- Weather ----------------
def test_weather_forecast(client, monkeypatch):
    monkeypatch.setattr(weather_api, "fetch_open_meteo_forecast", fake_weather)
    res = client.get("/api/weather/forecast?latitude=12.5266&longitude=78.2147")
    assert res.status_code == 200
    data = res.json()
    assert data["current_temperature"] == 27.0
    assert len(data["forecast_days"]) == 7
    first = data["forecast_days"][0]
    assert first["hours_rh95_t25_30"] == 14 and first["anthracnose_risk"] == "High"
    assert first["disease_spread_risk"] == "High" and data["pathogen_alerts"]
    assert data["forecast_days"][1]["anthracnose_risk"] == "Unknown"      # no hourly data for that day
    assert data["forecast_days"][6]["temperature_max"] is None             # null stays null


def test_weather_offline_returns_503_not_fake_data(client, monkeypatch):
    monkeypatch.setattr(weather_api, "fetch_open_meteo_forecast", offline_weather)
    assert client.get("/api/weather/forecast").status_code == 503


# ---------------- Farms ----------------
def test_farm_management(client, auth):
    farms = client.get("/api/farm/", headers=auth).json()
    assert farms[0]["name"] == "Krishnagiri Mango Valley"

    created = client.post("/api/farm/", headers=auth, json={
        "name": "Dharmapuri Alphonso Orchard", "latitude": 12.1211, "longitude": 78.1582,
        "cultivar": "Alphonso", "acreage": 10.0, "tree_count": 280,
    })
    assert created.status_code == 201
    farm_id = created.json()["id"]

    updated = client.put(f"/api/farm/{farm_id}", headers=auth, json={"acreage": 11.5})
    assert updated.json()["acreage"] == 11.5
    assert updated.json()["name"] == "Dharmapuri Alphonso Orchard"

    tree = client.post(f"/api/farm/{farm_id}/trees", headers=auth,
                       json={"tag_code": "DP-R01-T01", "row_number": 1, "tree_number": 1})
    assert tree.status_code == 201
    assert len(client.get(f"/api/farm/{farm_id}/trees", headers=auth).json()) == 1

    assert client.post("/api/farm/", headers=auth, json={"name": "Bad", "latitude": 95, "longitude": 0}).status_code == 422
    assert client.get("/api/farm/farm-missing", headers=auth).status_code == 404


def test_analyze_uses_farm_coordinates(client, auth):
    farm = client.get("/api/farm/", headers=auth).json()[0]
    data = _analyze(client, auth, farm_id=farm["id"]).json()
    assert (data["latitude"], data["longitude"]) == (farm["latitude"], farm["longitude"])
    assert data["farm_id"] == farm["id"]


# ---------------- Chatbot ----------------
def test_chatbot_english_answers_come_from_cited_database(client):
    data = client.post("/api/chatbot/", json={"message": "What is the treatment for Anthracnose?"}).json()
    assert "copper oxychloride" in data["reply"].lower()
    assert any("TNAU" in s for s in data["sources"])
    assert all(u.startswith("https://agritech.tnau.ac.in/") for u in data["source_urls"])
    assert data["matched_conditions"] == ["Anthracnose"]

    mildew = client.post("/api/chatbot/", json={"message": "powdery mildew on panicles"}).json()
    assert "sulphur" in mildew["reply"].lower()

    hopper = client.post("/api/chatbot/", json={"message": "mango hopper control"}).json()
    assert "unverified" in hopper["reply"] and "Imidacloprid" in hopper["reply"]


def test_chatbot_tamil(client):
    data = client.post("/api/chatbot/", json={"message": "ஆந்த்ராக்னோஸ் இலை நோய் தீர்வு என்ன?", "language": "ta"}).json()
    assert data["language"] == "ta"
    assert data["matched_conditions"] == ["Anthracnose"]
    assert "ஆந்த்ராக்னோஸ்" in data["reply"] and "Copper oxychloride" in data["reply"]


def test_chatbot_validates_input(client):
    assert client.post("/api/chatbot/", json={"message": "", "language": "en"}).status_code == 422
    assert client.post("/api/chatbot/", json={"message": "hi", "language": "fr"}).status_code == 422


def test_model_registry_endpoint(client):
    data = client.get("/api/models").json()
    assert isinstance(data, dict)
    for entry in data.values():
        assert entry["status"] in ("ok", "missing", "sha_mismatch") and entry["classes"]
