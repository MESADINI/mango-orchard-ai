"""
AI Mango Orchard Health & Pest Intelligence — Analyze Endpoint
Connects uploaded leaf/canopy imagery to the PyTorch, Grad-CAM, YOLOv8, and OHI engines.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict

from app.core.config import settings
from app.core.deps import get_current_user, get_store
from app.core.store import Store
from app.services.pipeline_service import InvalidImageError, pipeline_service

logger = logging.getLogger(__name__)
router = APIRouter()


class DiseaseDetectionResult(BaseModel):
    disease_name: str
    confidence: float
    severity_percentage: float
    is_healthy: bool
    gradcam_overlay_url: Optional[str] = None
    probabilities: Dict[str, float] = {}


class PestDetectionResult(BaseModel):
    pest_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    is_unreliable: bool  # True for Mango Hopper & Stem Borer (<100 annotated samples)
    warning: Optional[str] = None


class RecommendationItem(BaseModel):
    condition: str = ""
    treatment_type: str
    title: str
    product_name: str
    active_ingredient: str
    dosage_rate: str
    application_guidance: str
    waiting_period_days: int
    citation_source: str
    citation_url: str


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    scan_id: str
    created_at: str
    scan_type: str
    farm_id: Optional[str] = None
    tree_id: Optional[str] = None
    latitude: float
    longitude: float
    model_status: Dict[str, bool]
    ohi_score: float
    health_status: str
    badge_color: str
    ohi_is_partial: bool
    ohi_missing_components: List[str]
    lesion_severity_percentage: Optional[float] = None
    disease_results: List[DiseaseDetectionResult]
    pest_results: List[PestDetectionResult]
    unreliable_pest_warning: Optional[str] = None
    recommendations: List[RecommendationItem]
    nutrient_input_was_defaulted: bool = True
    caveats: List[str]
    penalty_breakdown: Dict[str, float]
    weights_used: Dict[str, float]
    weather_vulnerability: Dict[str, Any]
    image_url: Optional[str] = None
    pest_image_url: Optional[str] = None
    pest_overlay_url: Optional[str] = None


def _read_upload(upload: Optional[UploadFile]) -> Optional[bytes]:
    if upload is None:
        return None
    # Read at most one byte past the limit so oversized files are rejected without loading them fully.
    return upload.file.read(settings.max_upload_bytes + 1)


@router.post("/", response_model=AnalysisResponse)
def analyze_image(
    file: Optional[UploadFile] = File(None, description="Mango leaf image (JPEG/PNG/WebP)"),
    pest_file: Optional[UploadFile] = File(None, description="Optional canopy / pest trap image"),
    scan_type: Literal["dual", "disease", "pest"] = Form("dual"),
    farm_id: Optional[str] = Form(None),
    tree_id: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None, ge=-90, le=90, description="Orchard latitude for microclimate lookups"),
    longitude: Optional[float] = Form(None, ge=-180, le=180, description="Orchard longitude for microclimate lookups"),
    user=Depends(get_current_user),
    store: Store = Depends(get_store),
):
    """
    Analyzes uploaded mango orchard imagery:
    - Disease Detection (EfficientNet-B3) + Grad-CAM heatmap + lesion surface area %
    - Pest Detection (YOLOv8) with Rule 3 unreliable-class flagging
    - Open-Meteo microclimate, deterministic OHI and ICAR/TNAU recommendations

    Declared as a sync endpoint so FastAPI runs the CPU-bound inference in its threadpool.
    """
    if scan_type in ("dual", "disease") and file is None:
        raise HTTPException(status_code=400, detail="A leaf image ('file') is required for disease or dual scans.")
    if scan_type == "pest" and file is None and pest_file is None:
        raise HTTPException(status_code=400, detail="Upload a pest/canopy image ('pest_file' or 'file').")

    if farm_id:
        farm = store.get_farm(user["id"], farm_id)
        if not farm:
            raise HTTPException(status_code=404, detail="Farm not found.")
        if latitude is None or longitude is None:
            latitude, longitude = farm["latitude"], farm["longitude"]

    try:
        result = pipeline_service.analyze_scan(
            leaf_bytes=_read_upload(file),
            pest_bytes=_read_upload(pest_file),
            latitude=latitude,
            longitude=longitude,
            scan_type=scan_type,
            farm_id=farm_id,
            tree_id=tree_id,
        )
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Inference pipeline failed")
        raise HTTPException(status_code=500, detail="Inference pipeline failed. See server logs for details.")

    store.save_scan(user["id"], result)
    return AnalysisResponse(**result)


@router.get("/", response_model=List[AnalysisResponse])
def list_scans(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(get_current_user),
    store: Store = Depends(get_store),
):
    """Lists the current user's scans, newest first."""
    return [AnalysisResponse(**s) for s in store.list_scans(user["id"], limit=limit, offset=offset)]


@router.get("/{scan_id}", response_model=AnalysisResponse)
def get_scan_by_id(scan_id: str, user=Depends(get_current_user), store: Store = Depends(get_store)):
    """Retrieves a stored diagnosis and OHI result by scan ID."""
    scan = store.get_scan(user["id"], scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail=f"Scan ID {scan_id} not found.")
    return AnalysisResponse(**scan)
