"""
AI Mango Orchard Health & Pest Intelligence — Pest Detection Inference
Architecture: YOLO (Ultralytics); the deployed model is chosen by ml/pest_detection/train.py.
Enforces UNRELIABLE_CLASSES detection flags for Mango Hopper and Stem Borer, and marks every class whose training
data did not come from mango (ml/datasets/pest_source_review.json) as host_verified = False.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from PIL import Image, ImageDraw
import numpy as np

logger = logging.getLogger(__name__)

# Inviolable Architectural Rule 3:
# Mango Hopper and Stem Borer must NOT be presented as reliable detections
# without >= 100 labeled bounding boxes. They are tagged as UNRELIABLE_CLASSES.
UNRELIABLE_CLASSES = ["Mango Hopper", "Stem Borer"]

CONFIRMED_CLASSES = [
    "Fruit Fly",
    "Mealy Bug",
    "Scale Insect"
]

UNRELIABLE_WARNING = "Sample size < 100 images; detection is marked unverified"
HOST_UNVERIFIED_WARNING = ("Trained on images from other host plants or traps, not mango; detection is marked "
                           "unverified")
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "pest_detector.pt"


def load_host_verified(weights_path: Union[str, Path]) -> Dict[str, bool]:
    """host_verified per class from the metadata written next to the weights at deployment; {} if absent."""
    meta = Path(weights_path).with_suffix(".json")
    if not meta.is_file():
        return {}
    return {str(k): bool(v) for k, v in json.loads(meta.read_text(encoding="utf-8")).get("host_verified", {}).items()}


def build_detection(pest_name: str, confidence: float, bbox: List[float],
                    host_verified: bool = False) -> Dict[str, Any]:
    """Builds a detection record with the Rule 3 reliability flag (by class name) and the host-plant flag.
    A class is host-verified only when the deployment metadata says so; the default is False."""
    is_unreliable = pest_name in UNRELIABLE_CLASSES
    warning = UNRELIABLE_WARNING if is_unreliable else (None if host_verified else HOST_UNVERIFIED_WARNING)
    return {
        "pest_name": pest_name,
        "confidence": round(float(confidence), 4),
        "bbox": [float(v) for v in bbox],
        "is_unreliable": is_unreliable,
        "host_verified": bool(host_verified),
        "warning": warning,
    }


class PestDetector:
    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = str(weights_path) if weights_path else str(DEFAULT_WEIGHTS_PATH)
        self.model = None
        self.host_verified = load_host_verified(self.weights_path)
        if Path(self.weights_path).is_file():
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights_path)
                logger.info("Loaded YOLO weights from %s", self.weights_path)
            except Exception as e:
                logger.error("Failed to load YOLO model from %s: %s", self.weights_path, e)
        else:
            logger.warning(
                "Pest model weights not found (%s). Pest detection is disabled until a model "
                "is trained with ml/pest_detection/train.py.", self.weights_path
            )

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    @property
    def has_host_verified_classes(self) -> bool:
        """True if at least one class was trained on mango images. Without one, "no pest found" is not evidence
        of a pest-free mango tree, so the pipeline excludes the pest component from the OHI."""
        return any(self.host_verified.get(c, False) for c in CONFIRMED_CLASSES)

    def predict(
        self,
        image: Union[Image.Image, str, np.ndarray, Path],
        conf_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 detection and strictly flags UNRELIABLE_CLASSES.
        Returns bounding boxes, confidence, class names, and reliability flags.
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.convert("RGB")

        detections: List[Dict[str, Any]] = []
        if self.model is None:
            return detections

        results = self.model(pil_image, conf=conf_threshold, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                detections.append(build_detection(
                    cls_name, box.conf[0].item(), box.xyxy[0].tolist(),
                    host_verified=self.host_verified.get(cls_name, False),
                ))
        return detections

    def draw_detections(
        self,
        image: Union[Image.Image, str, np.ndarray, Path],
        detections: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Draws bounding boxes and labels onto the image.
        Confirmed classes are highlighted in green.
        Unreliable classes (Rule 3) are marked with high-visibility amber/warning badges.
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.copy())
        else:
            pil_image = image.copy()

        draw = ImageDraw.Draw(pil_image)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["pest_name"]
            conf = det["confidence"]
            is_unreliable = det.get("is_unreliable", False) or not det.get("host_verified", True)

            if is_unreliable:
                box_color = (234, 88, 12)    # Amber / Orange-600
                text_bg = (194, 65, 12)       # Dark amber
                label = f"! {cls_name} [UNVERIFIED] ({conf*100:.1f}%)"
            else:
                box_color = (34, 197, 94)     # Emerald / Green-500
                text_bg = (22, 101, 52)       # Dark forest green
                label = f"{cls_name} ({conf*100:.1f}%)"

            # Draw outer rectangle
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            # Draw label banner
            text_width = len(label) * 7 + 10
            text_height = 18
            draw.rectangle([x1, max(0, y1 - text_height), x1 + text_width, y1], fill=text_bg)
            draw.text((x1 + 4, max(0, y1 - text_height) + 2), label, fill=(255, 255, 255))

            if is_unreliable and det.get("warning"):
                warn_msg = det["warning"]
                warn_width = len(warn_msg) * 6 + 10
                draw.rectangle([x1, y2, x1 + warn_width, y2 + 16], fill=(120, 53, 15))
                draw.text((x1 + 4, y2 + 2), warn_msg, fill=(254, 240, 138))

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            pil_image.save(out, "JPEG", quality=95)

        return np.array(pil_image)
