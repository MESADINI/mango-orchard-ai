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

Any component that cannot run (missing model weights, an uncertain prediction,
weather service offline) is reported explicitly in `model_status`, in the OHI
`missing_components` and in the caveats. It is excluded from the OHI, never
replaced by a default such as "Healthy" or neutral weather.
"""

import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ml.disease_detection.inference import DiseaseDetector
from ml.explainability.gradcam import HookBasedGradCAM
from ml.forecasting.forecaster import forecast_disease_spread
from ml.forecasting.recommendations import recommendation_engine
from ml.forecasting.weather import fetch_open_meteo_forecast
from ml.pest_detection.inference import PestDetector
from ml.severity.ohi import OrchardHealthIndexEngine
from ml.severity.severity import compute_lesion_severity

logger = logging.getLogger(__name__)

# Default to major mango production cluster: Krishnagiri, Tamil Nadu
DEFAULT_LATITUDE = 12.5266
DEFAULT_LONGITUDE = 78.2147

DEFAULT_DISEASE_WEIGHTS = REPO_DIR / "ml" / "disease_detection" / "weights" / "disease_classifier.pth"
DEFAULT_PEST_WEIGHTS = REPO_DIR / "ml" / "pest_detection" / "weights" / "pest_detector.pt"
DEFAULT_ARTIFACTS_DIR = REPO_DIR / "ml" / "artifacts"

SCAN_TYPES = ("dual", "disease", "pest")

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return img.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class MangoOrchardPipeline:
    def __init__(
        self,
        disease_weights: Optional[str] = None,
        pest_weights: Optional[str] = None,
        weather_fetcher: Callable[[float, float], Optional[Dict[str, Any]]] = fetch_open_meteo_forecast,
        artifacts_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.disease_detector = DiseaseDetector(weights_path=disease_weights or str(DEFAULT_DISEASE_WEIGHTS),
                                                device=device)
        self.pest_detector = PestDetector(weights_path=pest_weights or str(DEFAULT_PEST_WEIGHTS))
        self.ohi_engine = OrchardHealthIndexEngine()
        self.weather_fetcher = weather_fetcher
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
        # Grad-CAM registers hooks on the shared model, so analyses are serialised.
        self._lock = threading.Lock()

    @property
    def model_status(self) -> Dict[str, bool]:
        return {
            "disease_model_trained": self.disease_detector.is_trained,
            "pest_model_trained": self.pest_detector.is_trained,
        }

    def _generate_gradcam(self, pil_leaf: Image.Image, output_path: Path) -> Optional[str]:
        model = self.disease_detector.model
        target_layer = getattr(model, "conv_head", None) if model is not None else None
        if target_layer is None:
            return None

        cam = HookBasedGradCAM(model, target_layer=target_layer)
        try:
            input_tensor = self.disease_detector.transform(pil_leaf).unsqueeze(0).to(self.disease_detector.device)
            heatmap, _ = cam.generate_heatmap(input_tensor)
            overlay_np = cam.overlay_on_image(np.array(pil_leaf), heatmap)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay_np).save(output_path, "JPEG", quality=95)
            return str(output_path)
        except Exception:
            logger.exception("Grad-CAM generation failed")
            return None
        finally:
            cam.remove_hooks()

    def _fetch_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        data = None
        try:
            data = self.weather_fetcher(latitude, longitude)
        except Exception:
            logger.exception("Weather fetch failed")

        current = (data or {}).get("current") or {}
        temp = _float_or_none(current.get("temperature_2m"))
        rh = _float_or_none(current.get("relative_humidity_2m"))
        precip = _float_or_none(current.get("precipitation"))
        live = None not in (temp, rh, precip)
        daily = (data or {}).get("daily")
        forecasts = forecast_disease_spread(daily, (data or {}).get("hourly")) if daily else []

        return {
            "live_data": live,
            "current_temperature_c": temp,
            "current_relative_humidity_pct": rh,
            "current_precipitation_mm": precip,
            "7_day_forecast": forecasts[:7],
        }

    def analyze_orchard(
        self,
        leaf_image: Optional[ImageInput],
        pest_image: Optional[ImageInput] = None,
        latitude: float = DEFAULT_LATITUDE,
        longitude: float = DEFAULT_LONGITUDE,
        generate_gradcam: bool = True,
        pest_conf_threshold: float = 0.25,
        scan_type: str = "dual",
        output_dir: Optional[Union[str, Path]] = None,
        output_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a multi-modal orchard diagnostic analysis.

        scan_type:
            "disease" - leaf disease, Grad-CAM and lesion severity only
            "pest"    - pest detection only (on pest_image, or leaf_image if absent)
            "dual"    - both
        Overlay images are written to `output_dir` (default ml/artifacts) using
        `output_prefix` so concurrent scans never overwrite each other.
        """
        if scan_type not in SCAN_TYPES:
            raise ValueError(f"scan_type must be one of {SCAN_TYPES}, got {scan_type!r}")
        run_disease = scan_type in ("dual", "disease")
        run_pest = scan_type in ("dual", "pest")
        if run_disease and leaf_image is None:
            raise ValueError("A leaf image is required for disease analysis.")
        if run_pest and pest_image is None and leaf_image is None:
            raise ValueError("An image is required for pest analysis.")

        out_dir = Path(output_dir) if output_dir else self.artifacts_dir
        prefix = output_prefix or uuid.uuid4().hex
        caveats: List[str] = []

        with self._lock:
            # --- Disease detection, Grad-CAM, lesion severity ---
            disease_name = None
            disease_conf = 0.0
            low_confidence = False
            probabilities: Dict[str, float] = {}
            severity_result: Dict[str, Any] = {"severity_percentage": 0.0, "lesion_pixels": 0, "total_leaf_pixels": 0}
            gradcam_overlay_path = None

            if run_disease:
                pil_leaf = _to_pil(leaf_image)
                disease_pred = self.disease_detector.predict(pil_leaf)
                disease_name = disease_pred["disease_name"]
                disease_conf = disease_pred["confidence"]
                probabilities = disease_pred.get("all_probabilities", {})
                low_confidence = bool(disease_pred.get("low_confidence", False))
                severity_result = compute_lesion_severity(np.array(pil_leaf))

                if self.disease_detector.is_trained:
                    if generate_gradcam:
                        gradcam_overlay_path = self._generate_gradcam(pil_leaf, out_dir / f"{prefix}_gradcam.jpg")
                    if low_confidence:
                        caveats.append(
                            f"The disease model is unsure about this photo (top confidence {disease_conf:.2f}). "
                            "It may not be a clear photo of a mango leaf, or the disease may be one the model does "
                            "not know. Please retake the photo: one leaf, filling the frame, in daylight. The disease "
                            "component is excluded from the OHI and no treatment is suggested."
                        )
                else:
                    caveats.append(
                        "Disease classification unavailable: no trained disease model is installed. "
                        "The disease component is excluded from the OHI."
                    )

            # --- Pest detection ---
            pest_detections: List[Dict[str, Any]] = []
            pest_overlay_path = None
            if run_pest:
                pest_input = pest_image if pest_image is not None else leaf_image
                if self.pest_detector.is_trained:
                    pil_pest = _to_pil(pest_input)
                    pest_detections = self.pest_detector.predict(pil_pest, conf_threshold=pest_conf_threshold)
                    if pest_detections:
                        pest_out = out_dir / f"{prefix}_pests.jpg"
                        self.pest_detector.draw_detections(pil_pest, pest_detections, output_path=pest_out)
                        pest_overlay_path = str(pest_out)
                else:
                    caveats.append(
                        "Pest detection unavailable: no trained YOLOv8 weights are installed. "
                        "The pest component is excluded from the OHI."
                    )

        confirmed_pests = [d for d in pest_detections
                           if not d.get("is_unreliable", False) and d.get("host_verified", True) is not False]
        unreliable_pests = [d for d in pest_detections if d.get("is_unreliable", False)]
        host_unverified_pests = [d for d in pest_detections
                                 if not d.get("is_unreliable", False) and d.get("host_verified", True) is False]
        pest_ok = run_pest and self.pest_detector.is_trained and self.pest_detector.has_host_verified_classes
        if run_pest and self.pest_detector.is_trained and not pest_ok:
            caveats.append(
                "The pest detector was trained only on non-mango images (traps, grape, coffee). Its detections are "
                "shown as unverified, and the pest component is excluded from the OHI."
            )

        # --- Weather & microclimate ---
        weather = self._fetch_weather(latitude, longitude)
        if not weather["live_data"]:
            caveats.append(
                "Live weather data unavailable (Open-Meteo unreachable); the weather component is excluded "
                "from the OHI."
            )

        # --- Deterministic OHI: components that were not assessed are excluded, never assumed healthy ---
        disease_ok = run_disease and self.disease_detector.is_trained and not low_confidence
        missing = [c for c, ok in (("disease", disease_ok),
                                   ("pest", pest_ok),
                                   ("weather", weather["live_data"])) if not ok]
        ohi_result = self.ohi_engine.compute_ohi(
            disease_name=disease_name if disease_ok else None,
            disease_confidence=disease_conf,
            disease_severity_pct=severity_result.get("severity_percentage", 0.0),
            pest_detections=pest_detections,
            relative_humidity=weather["current_relative_humidity_pct"],
            temperature=weather["current_temperature_c"],
            precipitation_mm=weather["current_precipitation_mm"],
            missing=missing,
        )
        ohi_result["caveats"] = list(ohi_result.get("caveats", [])) + caveats

        # --- Verified ICAR/TNAU recommendations (confirmed detections only) ---
        rec_disease = disease_name if disease_ok else None
        recommendations = recommendation_engine.get_recommendations_for_scan(
            disease_name=rec_disease,
            pest_names=sorted({d["pest_name"] for d in confirmed_pests}),
        )

        return {
            "status": "success",
            "scan_type": scan_type,
            "model_status": self.model_status,
            "location": {"latitude": latitude, "longitude": longitude},
            "disease_diagnosis": {
                "performed": run_disease,
                "model_available": self.disease_detector.is_trained,
                "disease_name": disease_name,
                "confidence": disease_conf,
                "low_confidence": low_confidence,
                "probabilities": probabilities,
                "lesion_severity_percentage": severity_result.get("severity_percentage", 0.0),
                "leaf_pixels": severity_result.get("total_leaf_pixels", 0),
                "lesion_pixels": severity_result.get("lesion_pixels", 0),
                "gradcam_overlay_path": gradcam_overlay_path,
            },
            "pest_intelligence": {
                "performed": run_pest,
                "model_available": self.pest_detector.is_trained,
                "total_detections": len(pest_detections),
                "confirmed_count": len(confirmed_pests),
                "unreliable_count": len(unreliable_pests),
                "host_unverified_count": len(host_unverified_pests),
                "model_host_verified": self.pest_detector.has_host_verified_classes,
                "detections": pest_detections,
                "pest_overlay_path": pest_overlay_path,
                "has_unreliable_classes": len(unreliable_pests) > 0,
                "unreliable_warning": (
                    "One or more detected pests (Mango Hopper / Stem Borer) are flagged as unverified "
                    "(training sample size < 100 images per Inviolable Rule 3)."
                ) if unreliable_pests else None,
            },
            "weather_vulnerability": weather,
            "orchard_health_index": ohi_result,
            "treatment_protocols": {
                "total_protocols": len(recommendations),
                "source_citations_guaranteed": True,
                "protocols": recommendations,
            },
        }
