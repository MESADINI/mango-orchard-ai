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

    def test_missing_components_are_excluded_and_weights_renormalised(self):
        res = self.engine.compute_ohi(disease_name="Anthracnose", disease_confidence=1.0,
                                      disease_severity_pct=0.0, missing=["pest", "weather"])
        self.assertEqual(res["weights_used"]["disease"], 1.0)
        self.assertEqual(res["ohi_score"], 60.0)                  # 100 - 1.0 * (1.00 * 40)
        self.assertEqual(res["missing_components"], ["pest", "weather"])
        self.assertTrue(res["is_partial"])
        self.assertIsNone(res["penalty_breakdown"]["pest_pressure_penalty"])

    def test_weights_always_sum_to_one(self):
        for missing in ([], ["pest"], ["weather"], ["disease", "weather"]):
            res = self.engine.compute_ohi(missing=missing)
            self.assertAlmostEqual(sum(res["weights_used"].values()), 1.0, places=3)
        res = self.engine.compute_ohi(nutrient_deficiency_score=20.0)
        self.assertEqual(res["weights_used"], {"disease": 0.4, "pest": 0.35, "weather": 0.15, "nutrient": 0.1})
        self.assertFalse(res["nutrient_input_was_defaulted"])

    def test_no_image_component_means_no_score(self):
        res = self.engine.compute_ohi(disease_name=None, missing=["pest"])
        self.assertIsNone(res["ohi_score"])
        self.assertTrue(res["health_status"].startswith("Unknown"))
        self.assertIsNone(res["penalty_breakdown"]["weighted_total_penalty"])

    def test_missing_disease_is_not_healthy(self):
        # same weather; a missing disease component must not score like a healthy leaf
        healthy = self.engine.compute_ohi(disease_name="Healthy", relative_humidity=90, missing=["pest"])
        missing = self.engine.compute_ohi(disease_name=None, relative_humidity=90, missing=["pest"])
        self.assertIsNotNone(healthy["ohi_score"])
        self.assertIsNone(missing["ohi_score"])

    def test_unreliable_pests_are_excluded(self):
        dets = [{"pest_name": "Fruit Fly", "confidence": 1.0},
                {"pest_name": "Stem Borer", "confidence": 1.0, "is_unreliable": True}]
        res = self.engine.compute_ohi(pest_detections=dets)
        self.assertEqual(res["penalty_breakdown"]["pest_pressure_penalty"], 18.0)
        self.assertEqual(res["excluded_unverified_pest_detections"], 1)

    def test_healthy_with_lesions_is_flagged(self):
        res = self.engine.compute_ohi(disease_name="Healthy", disease_severity_pct=12.0)
        self.assertTrue(res["healthy_lesion_inconsistency"])
        self.assertEqual(res["penalty_breakdown"]["disease_severity_penalty"], 0.0)
        self.assertTrue(any("retake" in c for c in res["caveats"]))
        self.assertFalse(self.engine.compute_ohi(disease_name="Healthy", disease_severity_pct=1.0)
                         ["healthy_lesion_inconsistency"])

    def test_unknown_classes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.compute_ohi(disease_name="Leaf Spot")
        with self.assertRaises(ValueError):
            self.engine.compute_ohi(pest_detections=[{"pest_name": "Aphid", "confidence": 1.0}])
        with self.assertRaises(ValueError):
            self.engine.compute_ohi(missing=["soil"])


if __name__ == "__main__":
    unittest.main()



def test_host_unverified_pest_detections_are_excluded():
    dets = [{"pest_name": "Fruit Fly", "confidence": 0.9, "is_unreliable": False, "host_verified": False},
            {"pest_name": "Mealy Bug", "confidence": 0.9, "is_unreliable": False, "host_verified": True}]
    r = OrchardHealthIndexEngine().compute_ohi("Healthy", 0.9, 0.0, dets, 60.0, 25.0, 0.0)
    assert r["excluded_unverified_pest_detections"] == 1
    only_mealy = OrchardHealthIndexEngine().compute_ohi("Healthy", 0.9, 0.0, dets[1:], 60.0, 25.0, 0.0)
    assert r["penalty_breakdown"] == only_mealy["penalty_breakdown"]
    assert any("non-mango" in c for c in r["caveats"])


def test_methodology_worked_example():
    """docs/OHI_METHODOLOGY.md section 5 must match the engine."""
    fly = {"pest_name": "Fruit Fly", "confidence": 0.8, "is_unreliable": False, "host_verified": True}
    r = OrchardHealthIndexEngine().compute_ohi("Anthracnose", 0.9, 20.0, [fly], 85.0, 26.0, 0.0)
    assert r["penalty_breakdown"]["disease_severity_penalty"] == 46.8
    assert r["penalty_breakdown"]["pest_pressure_penalty"] == 14.4
    assert r["penalty_breakdown"]["weather_vulnerability_index"] == 63.0
    assert r["penalty_breakdown"]["weighted_total_penalty"] == 36.9
    assert r["ohi_score"] == 63.1 and r["health_status"] == "High Infestation Risk"
