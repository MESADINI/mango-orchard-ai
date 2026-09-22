"""
AI Mango Orchard Health & Pest Intelligence — Backend API Test Suite
Validates all FastAPI routers using TestClient.
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend and repository root are on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from app.main import app

class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.leaf_path = REPO_DIR / "ml" / "datasets" / "sample_leaves" / "anthracnose_sample.jpg"
        cls.pest_path = REPO_DIR / "ml" / "datasets" / "sample_pests" / "fruit_fly_sample.jpg"
        cls.unreliable_pest_path = REPO_DIR / "ml" / "datasets" / "sample_pests" / "mango_hopper_sample.jpg"

    def test_01_health_check(self):
        """Validates API health and architectural metadata."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("models", data)
        self.assertIn("Rule 1: Deterministic OHI", data["models"]["rules_enforced"])
        self.assertIn("Rule 3: Unreliable Class Flagging", data["models"]["rules_enforced"])

    def test_02_analyze_image_endpoint(self):
        """Validates multi-modal scan execution with live PyTorch & YOLOv8 models."""
        self.assertTrue(self.leaf_path.exists(), f"Missing sample leaf at {self.leaf_path}")
        
        with open(self.leaf_path, "rb") as lf, open(self.pest_path, "rb") as pf:
            response = self.client.post(
                "/api/analyze/",
                files={
                    "file": ("anthracnose_sample.jpg", lf, "image/jpeg"),
                    "pest_file": ("fruit_fly_sample.jpg", pf, "image/jpeg")
                },
                data={
                    "scan_type": "dual",
                    "latitude": 12.5266,
                    "longitude": 78.2147
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        # OHI bounds and deterministic adherence
        self.assertIn("ohi_score", data)
        self.assertTrue(0.0 <= data["ohi_score"] <= 100.0)
        self.assertTrue(data["nutrient_input_was_defaulted"])
        self.assertIn("penalties", data.get("penalty_breakdown", {}) or data)

        # Disease Diagnosis & Grad-CAM
        self.assertGreaterEqual(len(data["disease_results"]), 1)
        d_res = data["disease_results"][0]
        self.assertEqual(d_res["disease_name"], "Anthracnose")
        self.assertIsNotNone(d_res["gradcam_overlay_url"])
        self.assertTrue(d_res["gradcam_overlay_url"].startswith("/media/gradcam/"))

        # Pest Detection
        self.assertIn("pest_results", data)

        # Source Citations
        self.assertGreater(len(data["recommendations"]), 0)
        for rec in data["recommendations"]:
            self.assertTrue(any(src in rec["citation_source"] for src in ["ICAR", "TNAU", "DPPQS", "Agritech"]))

    def test_03_unreliable_class_flagging(self):
        """Inviolable Rule 3: Gated classes (Mango Hopper) must be flagged is_unreliable=True."""
        with open(self.leaf_path, "rb") as lf, open(self.unreliable_pest_path, "rb") as pf:
            response = self.client.post(
                "/api/analyze/",
                files={
                    "file": ("anthracnose_sample.jpg", lf, "image/jpeg"),
                    "pest_file": ("mango_hopper_sample.jpg", pf, "image/jpeg")
                },
                data={"scan_type": "dual"}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        unreliable_found = any(p.get("is_unreliable") for p in data.get("pest_results", []))
        self.assertTrue(unreliable_found, "Expected Mango Hopper to be flagged as unreliable")

    def test_04_weather_forecast(self):
        """Validates Open-Meteo weather endpoint."""
        response = self.client.get("/api/weather/forecast?latitude=12.5266&longitude=78.2147")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("current_temperature", data)
        self.assertIn("forecast_days", data)
        self.assertGreater(len(data["forecast_days"]), 0)

    def test_05_pdf_report_generation(self):
        """Validates WeasyPrint executive PDF report generation."""
        response = self.client.get("/api/report/generate/test-scan-uuid-12345")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.headers.get("content-type"), ["application/pdf", "text/html; charset=utf-8", "text/html"])
        self.assertGreater(len(response.content), 100)

    def test_06_auth_flow(self):
        """Validates user login, registration, and JWT token issuance."""
        # 1. Login with developer credentials
        login_res = self.client.post(
            "/api/auth/login",
            json={"email": "grower@mango.ai", "password": "orchard123"}
        )
        self.assertEqual(login_res.status_code, 200)
        token = login_res.json()["access_token"]
        self.assertIsNotNone(token)

        # 2. Get profile with JWT
        me_res = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["email"], "grower@mango.ai")

    def test_07_farm_management(self):
        """Validates farm listing and registration."""
        # List farms
        list_res = self.client.get("/api/farm/")
        self.assertEqual(list_res.status_code, 200)
        farms = list_res.json()
        self.assertGreaterEqual(len(farms), 1)

        # Create new farm
        create_res = self.client.post(
            "/api/farm/",
            json={
                "name": "Dharmapuri Alphonso Orchard",
                "latitude": 12.1211,
                "longitude": 78.1582,
                "cultivar": "Alphonso",
                "acreage": 10.0,
                "tree_count": 280
            }
        )
        self.assertEqual(create_res.status_code, 200)
        self.assertEqual(create_res.json()["name"], "Dharmapuri Alphonso Orchard")

    def test_08_bilingual_chatbot(self):
        """Validates English and Tamil advisory responses with source citations."""
        # English query
        en_res = self.client.post(
            "/api/chatbot/",
            json={"message": "What is the treatment for Anthracnose?", "language": "en"}
        )
        self.assertEqual(en_res.status_code, 200)
        en_data = en_res.json()
        self.assertIn("copper oxychloride", en_data["reply"].lower())
        self.assertGreater(len(en_data["sources"]), 0)

        # Tamil query
        ta_res = self.client.post(
            "/api/chatbot/",
            json={"message": "ஆந்த்ராக்னோஸ் இலை நோய் தீர்வு என்ன?", "language": "ta"}
        )
        self.assertEqual(ta_res.status_code, 200)
        ta_data = ta_res.json()
        self.assertEqual(ta_data["language"], "ta")
        self.assertIn("காப்பர் ஆக்சிகுளோரைடு", ta_data["reply"])

if __name__ == "__main__":
    unittest.main()
