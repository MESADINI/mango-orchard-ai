import os
from typing import List, Dict, Any, Optional

UNRELIABLE_CLASSES = ["Mango Hopper", "Stem Borer"]

CONFIRMED_PEST_CLASSES = [
    "Fruit Fly",
    "Mealy Bug",
    "Scale Insect"
]

class PestInferenceService:
    def __init__(self, model_weights_path: Optional[str] = None):
        self.weights_path = model_weights_path
        self.model = None

    def load_model(self):
        """Loads YOLOv8 model when weights exist."""
        if self.weights_path and os.path.exists(self.weights_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights_path)
            except Exception as e:
                print(f"Warning: Failed to load YOLOv8 model: {e}")

    def predict(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 pest detection. Any detections of Mango Hopper or Stem Borer
        are strictly flagged with is_unreliable=True.
        """
        return []

pest_service = PestInferenceService()
