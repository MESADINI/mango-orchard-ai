"""
AI Mango Orchard Health & Pest Intelligence — Live Pipeline Service
Bridges FastAPI endpoints with the trained PyTorch, Grad-CAM, YOLOv8, and OHI engines.
"""

import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

from app.core.config import settings

# Ensure ML package is on sys.path
REPO_DIR = settings.REPO_DIR
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ml.pipeline import MangoOrchardPipeline

class PipelineService:
    def __init__(self):
        self.pipeline: Optional[MangoOrchardPipeline] = None
        self.scans_cache: Dict[str, Dict[str, Any]] = {}

    def get_pipeline(self) -> MangoOrchardPipeline:
        if self.pipeline is None:
            disease_weights = str(settings.DISEASE_WEIGHTS_PATH) if settings.DISEASE_WEIGHTS_PATH.exists() else None
            pest_weights = str(settings.PEST_WEIGHTS_PATH) if settings.PEST_WEIGHTS_PATH.exists() else None
            print(f"Initializing MangoOrchardPipeline with:\n  Disease: {disease_weights}\n  Pest: {pest_weights}")
            self.pipeline = MangoOrchardPipeline(disease_weights=disease_weights, pest_weights=pest_weights)
        return self.pipeline

    def analyze_scan(
        self,
        leaf_bytes: bytes,
        leaf_filename: str,
        pest_bytes: Optional[bytes] = None,
        pest_filename: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        scan_type: str = "dual",
        farm_id: Optional[str] = None,
        tree_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-modal scan using live ML pipelines.
        """
        scan_id = str(uuid.uuid4())
        lat = latitude if latitude is not None else settings.DEFAULT_LATITUDE
        lon = longitude if longitude is not None else settings.DEFAULT_LONGITUDE

        # 1. Save uploaded files to media directory preserving original file stems
        leaf_ext = Path(leaf_filename).suffix or ".jpg"
        leaf_stem = Path(leaf_filename).stem or "leaf"
        leaf_save_path = settings.MEDIA_DIR / "uploads" / f"{scan_id}_{leaf_stem}{leaf_ext}"
        with open(leaf_save_path, "wb") as f:
            f.write(leaf_bytes)

        pest_save_path = None
        if pest_bytes is not None:
            pest_ext = Path(pest_filename).suffix or ".jpg" if pest_filename else ".jpg"
            pest_stem = Path(pest_filename).stem if pest_filename else "pest"
            pest_save_path = settings.MEDIA_DIR / "uploads" / f"{scan_id}_{pest_stem}{pest_ext}"
            with open(pest_save_path, "wb") as f:
                f.write(pest_bytes)

        pipeline = self.get_pipeline()

        # 2. Execute analysis
        raw_result = pipeline.analyze_orchard(
            leaf_image=leaf_save_path,
            pest_image=pest_save_path if pest_save_path is not None else leaf_save_path,
            latitude=lat,
            longitude=lon,
            generate_gradcam=True,
            pest_conf_threshold=0.01
        )

        d_diag = raw_result["disease_diagnosis"]
        p_intel = raw_result["pest_intelligence"]
        ohi = raw_result["orchard_health_index"]
        weather = raw_result["weather_vulnerability"]
        protocols = raw_result["treatment_protocols"]["protocols"]

        # 3. Publish visual overlays to /media
        gradcam_url = None
        if d_diag.get("gradcam_overlay_path") and os.path.exists(d_diag["gradcam_overlay_path"]):
            target_gradcam = settings.MEDIA_DIR / "gradcam" / f"{scan_id}_gradcam.jpg"
            shutil.copy(d_diag["gradcam_overlay_path"], target_gradcam)
            gradcam_url = f"/media/gradcam/{scan_id}_gradcam.jpg"

        pest_url = None
        if p_intel.get("pest_overlay_path") and os.path.exists(p_intel["pest_overlay_path"]):
            target_pest = settings.MEDIA_DIR / "pests" / f"{scan_id}_pests.jpg"
            shutil.copy(p_intel["pest_overlay_path"], target_pest)
            pest_url = f"/media/pests/{scan_id}_pests.jpg"

        # 4. Map Pydantic compatible objects
        disease_results = [
            {
                "disease_name": d_diag["disease_name"],
                "confidence": round(d_diag["confidence"], 4),
                "severity_percentage": round(d_diag["lesion_severity_percentage"], 2),
                "is_healthy": d_diag["disease_name"] == "Healthy",
                "gradcam_overlay_url": gradcam_url
            }
        ]

        pest_results = []
        for det in p_intel["detections"]:
            pest_results.append({
                "pest_name": det["pest_name"],
                "confidence": round(det["confidence"], 4),
                "bbox": [round(v, 2) for v in det["bbox"]],
                "is_unreliable": det["is_unreliable"]
            })

        recommendations = []
        for r in protocols:
            recommendations.append({
                "treatment_type": r.get("treatment_type", "Cultural"),
                "title": r.get("title", "Management Advisory"),
                "product_name": r.get("product_name", "N/A"),
                "active_ingredient": r.get("active_ingredient", "N/A"),
                "dosage_rate": r.get("dosage_rate", "Follow label directions"),
                "application_guidance": r.get("application_guidance", "Apply as directed"),
                "waiting_period_days": r.get("waiting_period_days", 0),
                "citation_source": r.get("citation_source", "ICAR/TNAU"),
                "citation_url": r.get("citation_url", "https://agritech.tnau.ac.in")
            })

        # Compile final response dictionary
        penalties_dict = ohi.get("penalties", {})
        breakdown_dict = dict(ohi.get("penalty_breakdown", penalties_dict))
        breakdown_dict["penalties"] = penalties_dict

        response_data = {
            "scan_id": scan_id,
            "scan_type": scan_type,
            "farm_id": farm_id,
            "tree_id": tree_id,
            "ohi_score": ohi["ohi_score"],
            "health_status": ohi["health_status"],
            "disease_results": disease_results,
            "pest_results": pest_results,
            "recommendations": recommendations,
            "nutrient_input_was_defaulted": ohi["nutrient_input_was_defaulted"],
            "caveats": ohi["caveats"],
            "penalty_breakdown": breakdown_dict,
            "penalties": penalties_dict,
            "weather_vulnerability": weather,
            "image_url": f"/media/uploads/{scan_id}_{leaf_stem}{leaf_ext}",
            "pest_image_url": pest_url
        }

        # Cache scan in memory for PDF report generation and history
        self.scans_cache[scan_id] = response_data
        return response_data

    def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        return self.scans_cache.get(scan_id)

pipeline_service = PipelineService()
