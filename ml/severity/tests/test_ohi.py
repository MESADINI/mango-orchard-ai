"""Unit tests for Deterministic Orchard Health Index Engine."""

import os
import sys
from pathlib import Path
import unittest

# Ensure repo root is on sys.path for direct test execution
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.severity.ohi import OrchardHealthIndexEngine

class TestOHIEngine(unittest.TestCase):
    def setUp(self):
        self.engine = OrchardHealthIndexEngine()

    def test_perfect_healthy_tree(self):
        res = self.engine.compute_ohi(
            disease_name="Healthy",
            disease_confidence=1.0,
            disease_severity_pct=0.0,
            pest_detections=[],
            relative_humidity=50.0,
            temperature=20.0,
            precipitation_mm=0.0
        )
        self.assertTrue(res["ohi_score"] >= 95.0)
        self.assertEqual(res["health_status"], "Optimal Health")
        self.assertTrue(res["nutrient_input_was_defaulted"])

    def test_severe_disease_penalty(self):
        res = self.engine.compute_ohi(
            disease_name="Anthracnose",
            disease_confidence=0.95,
            disease_severity_pct=50.0,
            pest_detections=[
                {"pest_name": "Fruit Fly", "confidence": 0.90},
                {"pest_name": "Fruit Fly", "confidence": 0.85}
            ],
            relative_humidity=88.0,
            temperature=27.0,
            precipitation_mm=12.0
        )
        self.assertTrue(res["ohi_score"] < 50.0)
        self.assertEqual(res["health_status"], "Severe Outbreak / Critical")
        self.assertIn("caveats", res)
        self.assertEqual(len(res["caveats"]), 1)

    def test_bounds_adherence(self):
        # Even with extreme penalties, score should never drop below 0
        res = self.engine.compute_ohi(
            disease_name="Anthracnose",
            disease_confidence=1.0,
            disease_severity_pct=100.0,
            pest_detections=[{"pest_name": "Stem Borer", "confidence": 1.0} for _ in range(10)],
            relative_humidity=100.0,
            temperature=27.0,
            precipitation_mm=100.0
        )
        self.assertEqual(res["ohi_score"], 0.0)
        self.assertEqual(res["health_status"], "Severe Outbreak / Critical")
        self.assertGreaterEqual(res["ohi_score"], 0.0)
        self.assertLessEqual(res["ohi_score"], 100.0)

if __name__ == "__main__":
    unittest.main()

