"""
AI Mango Orchard Health & Pest Intelligence — Unified Diagnostic Pipeline
Integrates:
1. EfficientNet-B3 Disease Detection (PyTorch / timm)
2. Non-invasive Hook-Based Grad-CAM Explainability
3. HSV Lesion Surface Area Severity Segmentation
4. YOLOv8 Pest Detection with Unreliable Class Flagging (Rule 3)
5. Open-Meteo 7-Day Microclimate Ingestion
6. Epidemiological Disease Spread Risk Forecasting
7. Deterministic Orchard Health Index Engine (Rule 1 & Rule 4)
8. Source-Cited ICAR/TNAU/DPPQS Treatment Protocols (Rule 2)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import numpy as np

# Ensure repository root is on sys.path
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ml.disease_detection.inference import DiseaseDetector
from ml.explainability.gradcam import HookBasedGradCAM
from ml.severity.severity import compute_lesion_severity
from ml.pest_detection.inference import PestDetector
from ml.forecasting.weather import fetch_open_meteo_forecast
from ml.forecasting.forecaster import forecast_disease_spread
from ml.severity.ohi import OrchardHealthIndexEngine
from ml.forecasting.recommendations import recommendation_engine

# Default to major mango production cluster: Krishnagiri, Tamil Nadu
DEFAULT_LATITUDE = 12.5266
DEFAULT_LONGITUDE = 78.2147

class MangoOrchardPipeline:
    def __init__(
        self,
        disease_weights: Optional[str] = None,
        pest_weights: Optional[str] = None
    ):
        base_dir = Path(__file__).resolve().parent

        # 1. Disease Detector
        if disease_weights is None:
            candidate_disease = base_dir / "disease_detection" / "weights" / "efficientnet_b3_best.pth"
            disease_weights = str(candidate_disease) if candidate_disease.exists() else None
        self.disease_detector = DiseaseDetector(weights_path=disease_weights)

        # 2. Pest Detector
        self.pest_detector = PestDetector(weights_path=pest_weights)

        # 3. OHI Deterministic Engine
        self.ohi_engine = OrchardHealthIndexEngine()

        # Artifact output directories
        self.artifacts_dir = base_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "explainability" / "artifacts").mkdir(parents=True, exist_ok=True)
        (base_dir / "pest_detection" / "artifacts").mkdir(parents=True, exist_ok=True)

    def analyze_orchard(
        self,
        leaf_image: Union[str, Path, Image.Image, np.ndarray],
        pest_image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        latitude: float = DEFAULT_LATITUDE,
        longitude: float = DEFAULT_LONGITUDE,
        generate_gradcam: bool = True,
        pest_conf_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Executes complete, multi-modal orchard diagnostic analysis.
        """
        # --- [Step 1: Leaf Image Normalization] ---
        if isinstance(leaf_image, (str, Path)):
            pil_leaf = Image.open(leaf_image).convert("RGB")
        elif isinstance(leaf_image, np.ndarray):
            pil_leaf = Image.fromarray(leaf_image)
        else:
            pil_leaf = leaf_image.convert("RGB")
        leaf_np = np.array(pil_leaf)

        # --- [Step 2: Disease Detection (EfficientNet-B3)] ---
        disease_pred = self.disease_detector.predict(pil_leaf)
        disease_name = disease_pred["disease_name"]
        disease_conf = disease_pred["confidence"]

        # --- [Step 3: Hook-Based Grad-CAM Explainability] ---
        gradcam_overlay_path = None
        if generate_gradcam and self.disease_detector.model is not None:
            try:
                target_layer = getattr(self.disease_detector.model, "conv_head", None)
                if target_layer is not None:
                    cam = HookBasedGradCAM(self.disease_detector.model, target_layer=target_layer)
                    input_tensor = self.disease_detector.transform(pil_leaf).unsqueeze(0).to(self.disease_detector.device)
                    heatmap, _ = cam.generate_heatmap(input_tensor)
                    overlay_np = cam.overlay_on_image(leaf_np, heatmap)
                    
                    slug = disease_name.lower().replace(" ", "_")
                    gradcam_file = REPO_DIR / "ml" / "explainability" / "artifacts" / f"gradcam_{slug}_pipeline_overlay.jpg"
                    Image.fromarray(overlay_np).save(gradcam_file, "JPEG", quality=95)
                    gradcam_overlay_path = str(gradcam_file)
                    cam.remove_hooks()
            except Exception as e:
                print(f"Grad-CAM generation notice: {e}")

        # --- [Step 4: Lesion Severity Surface Segmentation] ---
        severity_result = compute_lesion_severity(leaf_np)
        disease_severity_pct = severity_result.get("severity_percentage", 0.0)

        # --- [Step 5: Pest Detection (YOLOv8 & Unreliable Gating)] ---
        pest_detections: List[Dict[str, Any]] = []
        pest_overlay_path = None
        target_pest_input = pest_image if pest_image is not None else leaf_image

        pest_detections = self.pest_detector.predict(target_pest_input, conf_threshold=pest_conf_threshold)
        if len(pest_detections) > 0:
            pest_out = REPO_DIR / "ml" / "pest_detection" / "artifacts" / "pipeline_pest_detections.jpg"
            self.pest_detector.draw_detections(target_pest_input, pest_detections, output_path=pest_out)
            pest_overlay_path = str(pest_out)

        # Separate confirmed vs unverified pest detections
        confirmed_pests = [d for d in pest_detections if not d.get("is_unreliable", False)]
        unreliable_pests = [d for d in pest_detections if d.get("is_unreliable", False)]

        # --- [Step 6: Weather & Microclimate Ingestion] ---
        weather_data = fetch_open_meteo_forecast(latitude=latitude, longitude=longitude)
        current_temp = 28.0
        current_rh = 70.0
        current_precip = 0.0
        disease_forecasts: List[Dict[str, Any]] = []

        if weather_data and "current" in weather_data:
            current_temp = float(weather_data["current"].get("temperature_2m", 28.0))
            current_rh = float(weather_data["current"].get("relative_humidity_2m", 70.0))
            current_precip = float(weather_data["current"].get("precipitation", 0.0))

        if weather_data and "daily" in weather_data:
            disease_forecasts = forecast_disease_spread(weather_data["daily"])

        # --- [Step 7: Deterministic Orchard Health Index (OHI)] ---
        ohi_result = self.ohi_engine.compute_ohi(
            disease_name=disease_name,
            disease_confidence=disease_conf,
            disease_severity_pct=disease_severity_pct,
            pest_detections=pest_detections,
            relative_humidity=current_rh,
            temperature=current_temp,
            precipitation_mm=current_precip
        )

        # --- [Step 8: Verified ICAR/TNAU Recommendations] ---
        pest_names_for_recs = [d["pest_name"] for d in confirmed_pests]
        recommendations = recommendation_engine.get_recommendations_for_scan(
            disease_name=disease_name,
            pest_names=pest_names_for_recs
        )

        return {
            "status": "success",
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "disease_diagnosis": {
                "disease_name": disease_name,
                "confidence": disease_conf,
                "probabilities": disease_pred.get("probabilities", {}),
                "lesion_severity_percentage": disease_severity_pct,
                "leaf_pixels": severity_result.get("leaf_pixels", 0),
                "lesion_pixels": severity_result.get("lesion_pixels", 0),
                "gradcam_overlay_path": gradcam_overlay_path
            },
            "pest_intelligence": {
                "total_detections": len(pest_detections),
                "confirmed_count": len(confirmed_pests),
                "unreliable_count": len(unreliable_pests),
                "detections": pest_detections,
                "pest_overlay_path": pest_overlay_path,
                "has_unreliable_classes": len(unreliable_pests) > 0,
                "unreliable_warning": (
                    "One or more detected pests (Mango Hopper / Stem Borer) are flagged as unverified "
                    "(training sample size < 100 images per Inviolable Rule 3)."
                ) if len(unreliable_pests) > 0 else None
            },
            "weather_vulnerability": {
                "current_temperature_c": current_temp,
                "current_relative_humidity_pct": current_rh,
                "current_precipitation_mm": current_precip,
                "7_day_forecast": disease_forecasts[:7]
            },
            "orchard_health_index": ohi_result,
            "treatment_protocols": {
                "total_protocols": len(recommendations),
                "source_citations_guaranteed": True,
                "protocols": recommendations
            }
        }
