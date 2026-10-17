"""
AI Mango Orchard Health & Pest Intelligence — Live Pipeline Service
Bridges FastAPI endpoints with the PyTorch, Grad-CAM, YOLOv8, and OHI engines.
"""

import io
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.store import utcnow

from ml.pipeline import MangoOrchardPipeline

logger = logging.getLogger(__name__)

# Saved extension is derived from the decoded image format, never from the client's filename.
ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class InvalidImageError(ValueError):
    pass


def validate_image(data: bytes, label: str) -> str:
    """Checks that `data` is a decodable JPEG/PNG/WebP image and returns the file extension to use."""
    if not data:
        raise InvalidImageError(f"The {label} file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise InvalidImageError(f"The {label} file exceeds the {settings.MAX_UPLOAD_MB} MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as e:
        raise InvalidImageError(f"The {label} file is not a valid image.") from e
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(f"The {label} file must be JPEG, PNG or WebP (got {fmt}).")
    return ALLOWED_FORMATS[fmt]


def media_url(path: Optional[str]) -> Optional[str]:
    """Converts a file path under MEDIA_DIR into its public /media URL."""
    if not path:
        return None
    rel = Path(path).resolve().relative_to(settings.MEDIA_DIR.resolve())
    return "/media/" + rel.as_posix()


class PipelineService:
    def __init__(self):
        self._pipeline: Optional[MangoOrchardPipeline] = None
        self._init_lock = threading.Lock()

    def get_pipeline(self) -> MangoOrchardPipeline:
        with self._init_lock:
            if self._pipeline is None:
                self._pipeline = MangoOrchardPipeline(
                    disease_weights=str(settings.DISEASE_WEIGHTS_PATH),
                    pest_weights=str(settings.PEST_WEIGHTS_PATH),
                )
            return self._pipeline

    def model_status(self) -> Dict[str, bool]:
        return self.get_pipeline().model_status

    def analyze_scan(
        self,
        leaf_bytes: Optional[bytes],
        pest_bytes: Optional[bytes] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        scan_type: str = "dual",
        farm_id: Optional[str] = None,
        tree_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validates and stores the uploads, runs the ML pipeline and returns the API response payload."""
        scan_id = str(uuid.uuid4())
        lat = latitude if latitude is not None else settings.DEFAULT_LATITUDE
        lon = longitude if longitude is not None else settings.DEFAULT_LONGITUDE

        uploads_dir = settings.MEDIA_DIR / "uploads"
        results_dir = settings.MEDIA_DIR / "results"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        leaf_path = pest_path = None
        if leaf_bytes is not None:
            leaf_path = uploads_dir / f"{scan_id}_leaf{validate_image(leaf_bytes, 'leaf image')}"
        if pest_bytes is not None:
            pest_path = uploads_dir / f"{scan_id}_pest{validate_image(pest_bytes, 'pest image')}"
        if leaf_path:
            leaf_path.write_bytes(leaf_bytes)
        if pest_path:
            pest_path.write_bytes(pest_bytes)

        raw = self.get_pipeline().analyze_orchard(
            leaf_image=leaf_path,
            pest_image=pest_path,
            latitude=lat,
            longitude=lon,
            generate_gradcam=True,
            pest_conf_threshold=settings.PEST_CONFIDENCE_THRESHOLD,
            scan_type=scan_type,
            output_dir=results_dir,
            output_prefix=scan_id,
        )
        return self._to_response(scan_id, raw, farm_id, tree_id, leaf_path, pest_path)

    @staticmethod
    def _to_response(scan_id, raw, farm_id, tree_id, leaf_path, pest_path) -> Dict[str, Any]:
        d_diag = raw["disease_diagnosis"]
        p_intel = raw["pest_intelligence"]
        ohi = raw["orchard_health_index"]

        disease_results = []
        if d_diag["performed"] and d_diag["model_available"]:
            disease_results.append({
                "disease_name": d_diag["disease_name"],
                "confidence": round(d_diag["confidence"], 4),
                "severity_percentage": round(d_diag["lesion_severity_percentage"], 2),
                "is_healthy": d_diag["disease_name"] == "Healthy",
                "gradcam_overlay_url": media_url(d_diag["gradcam_overlay_path"]),
                "probabilities": {k: round(v, 4) for k, v in d_diag["probabilities"].items()},
                "low_confidence": d_diag.get("low_confidence", False),
            })

        pest_results = [
            {
                "pest_name": det["pest_name"],
                "confidence": round(det["confidence"], 4),
                "bbox": [round(v, 2) for v in det["bbox"]],
                "is_unreliable": det["is_unreliable"],
                "host_verified": det.get("host_verified", False),
                "warning": det.get("warning"),
            }
            for det in p_intel["detections"]
        ]

        recommendations = [
            {
                "condition": r.get("condition", ""),
                "treatment_type": r.get("treatment_type", "Cultural"),
                "title": r.get("title", "Management Advisory"),
                "product_name": r.get("product_name", "N/A"),
                "active_ingredient": r.get("active_ingredient", "N/A"),
                "dosage_rate": r.get("dosage_rate", "Follow label directions"),
                "application_guidance": r.get("application_guidance", "Apply as directed"),
                "waiting_period_days": r.get("waiting_period_days"),
                "citation_source": r.get("citation_source", ""),
                "citation_url": r.get("citation_url", ""),
                "citation_quote": r.get("citation_quote", ""),
                "verified_on": r.get("verified_on", ""),
            }
            for r in raw["treatment_protocols"]["protocols"]
        ]

        return {
            "scan_id": scan_id,
            "created_at": utcnow(),
            "scan_type": raw["scan_type"],
            "farm_id": farm_id,
            "tree_id": tree_id,
            "latitude": raw["location"]["latitude"],
            "longitude": raw["location"]["longitude"],
            "model_status": raw["model_status"],
            "ohi_score": ohi["ohi_score"],
            "health_status": ohi["health_status"],
            "badge_color": ohi["badge_color"],
            "ohi_is_partial": ohi["is_partial"],
            "ohi_missing_components": ohi["missing_components"],
            "lesion_severity_percentage": (
                round(d_diag["lesion_severity_percentage"], 2) if d_diag["performed"] else None
            ),
            "disease_results": disease_results,
            "pest_results": pest_results,
            "unreliable_pest_warning": p_intel["unreliable_warning"],
            "recommendations": recommendations,
            "nutrient_input_was_defaulted": ohi["nutrient_input_was_defaulted"],
            "caveats": ohi["caveats"],
            "penalty_breakdown": ohi["penalty_breakdown"],
            "healthy_lesion_inconsistency": ohi["healthy_lesion_inconsistency"],
            "excluded_unverified_pest_detections": ohi["excluded_unverified_pest_detections"],
            "weights_used": ohi["weights_used"],
            "weather_vulnerability": raw["weather_vulnerability"],
            "image_url": media_url(str(leaf_path)) if leaf_path else None,
            "pest_image_url": media_url(str(pest_path)) if pest_path else None,
            "pest_overlay_url": media_url(p_intel["pest_overlay_path"]),
        }


pipeline_service = PipelineService()
