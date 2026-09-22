"""
AI Mango Orchard Health & Pest Intelligence — End-to-End Pipeline Integration Tests
Validates complete multi-modal integration, OHI bounds, and source citations.
"""

import sys
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure repository root is on sys.path
REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ml.pipeline import MangoOrchardPipeline

class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = MangoOrchardPipeline()
        cls.leaf_sample = REPO_DIR / "ml" / "datasets" / "sample_leaves" / "anthracnose_sample.jpg"
        cls.pest_sample = REPO_DIR / "ml" / "datasets" / "sample_pests" / "fruit_fly_sample.jpg"
        cls.unreliable_pest_sample = REPO_DIR / "ml" / "datasets" / "sample_pests" / "mango_hopper_sample.jpg"

    def test_pipeline_execution_schema(self):
        """Validates that analyze_orchard returns the full required multi-modal schema."""
        result = self.pipeline.analyze_orchard(
            leaf_image=self.leaf_sample,
            pest_image=self.pest_sample,
            latitude=12.5266,
            longitude=78.2147
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("disease_diagnosis", result)
        self.assertIn("pest_intelligence", result)
        self.assertIn("weather_vulnerability", result)
        self.assertIn("orchard_health_index", result)
        self.assertIn("treatment_protocols", result)

        # Disease Diagnosis assertions
        d_diag = result["disease_diagnosis"]
        self.assertIn("disease_name", d_diag)
        self.assertIn("confidence", d_diag)
        self.assertIn("lesion_severity_percentage", d_diag)

        # OHI assertions
        ohi = result["orchard_health_index"]
        self.assertTrue(0.0 <= ohi["ohi_score"] <= 100.0, f"OHI out of bounds: {ohi['ohi_score']}")
        self.assertTrue(ohi["nutrient_input_was_defaulted"], "Nutrient input must be disclosed as defaulted")
        self.assertIn("disease_severity_penalty", ohi["penalties"])
        self.assertIn("pest_pressure_penalty", ohi["penalties"])

    def test_unreliable_pest_pipeline_flagging(self):
        """Inviolable Rule 3: Gated pests must trigger system-wide unreliable flags."""
        result = self.pipeline.analyze_orchard(
            leaf_image=self.leaf_sample,
            pest_image=self.unreliable_pest_sample
        )
        p_intel = result["pest_intelligence"]
        self.assertTrue(p_intel["has_unreliable_classes"])
        self.assertIsNotNone(p_intel["unreliable_warning"])
        self.assertIn("sample size < 100", p_intel["unreliable_warning"].lower())

    def test_source_citations_enforcement(self):
        """Inviolable Rule 2: Treatment protocols must contain verified ICAR/TNAU/DPPQS citations."""
        result = self.pipeline.analyze_orchard(
            leaf_image=self.leaf_sample,
            pest_image=self.pest_sample
        )
        protocols = result["treatment_protocols"]["protocols"]
        self.assertGreater(len(protocols), 0, "Expected treatment recommendations for Anthracnose & Fruit Fly")
        
        for rec in protocols:
            citation = rec.get("citation_source", "")
            has_valid_source = any(src in citation for src in ["ICAR", "TNAU", "DPPQS", "Agritech"])
            self.assertTrue(has_valid_source, f"Recommendation lacks required official citation: {rec}")

if __name__ == "__main__":
    unittest.main()
