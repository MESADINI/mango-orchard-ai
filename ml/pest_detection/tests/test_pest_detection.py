"""
AI Mango Orchard Health & Pest Intelligence — Pest Detection Unit Tests
Validates YOLOv8 output structures and Inviolable Rule 3 (Unreliable Class Flagging).
"""

import os
import sys
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.pest_detection.inference import PestDetector, CONFIRMED_CLASSES, UNRELIABLE_CLASSES

class TestPestDetection(unittest.TestCase):
    def setUp(self):
        self.detector = PestDetector(weights_path="nonexistent_test_weights.pt")
        self.sample_img = Image.new("RGB", (640, 640), color=(100, 140, 80))

    def test_class_definitions(self):
        """Inviolable Rule 3: Confirmed and Unreliable classes must be properly segmented."""
        self.assertIn("Fruit Fly", CONFIRMED_CLASSES)
        self.assertIn("Mealy Bug", CONFIRMED_CLASSES)
        self.assertIn("Scale Insect", CONFIRMED_CLASSES)

        self.assertIn("Mango Hopper", UNRELIABLE_CLASSES)
        self.assertIn("Stem Borer", UNRELIABLE_CLASSES)
        
        # Ensure no overlap between confirmed and unreliable sets
        overlap = set(CONFIRMED_CLASSES).intersection(set(UNRELIABLE_CLASSES))
        self.assertEqual(len(overlap), 0, f"Overlap detected between confirmed and unreliable classes: {overlap}")

    def test_unreliable_class_gating_mango_hopper(self):
        """Inviolable Rule 3: Mango Hopper must be flagged with is_unreliable=True and warning."""
        sample_path = REPO_ROOT / "ml" / "datasets" / "sample_pests" / "mango_hopper_sample.jpg"
        
        detections = self.detector.predict(sample_path)
        self.assertGreaterEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det["pest_name"], "Mango Hopper")
        self.assertTrue(det["is_unreliable"], "Mango Hopper must be marked as unreliable!")
        self.assertIsNotNone(det["warning"])
        self.assertIn("Sample size < 100", det["warning"])

    def test_unreliable_class_gating_stem_borer(self):
        """Inviolable Rule 3: Stem Borer must be flagged with is_unreliable=True and warning."""
        sample_path = REPO_ROOT / "ml" / "datasets" / "sample_pests" / "stem_borer_sample.jpg"

        detections = self.detector.predict(sample_path)
        self.assertGreaterEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det["pest_name"], "Stem Borer")
        self.assertTrue(det["is_unreliable"], "Stem Borer must be marked as unreliable!")
        self.assertIsNotNone(det["warning"])
        self.assertIn("Sample size < 100", det["warning"])

    def test_draw_detections_overlay(self):
        """Tests that draw_detections renders bounding boxes and badges cleanly."""
        detections = [
            {
                "pest_name": "Fruit Fly",
                "confidence": 0.92,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "is_unreliable": False,
                "warning": None
            },
            {
                "pest_name": "Mango Hopper",
                "confidence": 0.85,
                "bbox": [300.0, 300.0, 420.0, 420.0],
                "is_unreliable": True,
                "warning": "Sample size < 100 images; detection is marked unverified"
            }
        ]

        out_path = Path(__file__).resolve().parent.parent / "artifacts" / "test_detection_overlay.jpg"
        annotated_np = self.detector.draw_detections(self.sample_img, detections, output_path=out_path)

        self.assertIsInstance(annotated_np, np.ndarray)
        self.assertEqual(annotated_np.shape, (640, 640, 3))
        self.assertTrue(out_path.exists())

if __name__ == "__main__":
    unittest.main()
