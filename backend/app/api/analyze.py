"""
AI Mango Orchard Health & Pest Intelligence — Analyze Endpoint
Connects uploaded leaf/canopy imagery to the live PyTorch, Grad-CAM, YOLOv8, and OHI engines.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.services.pipeline_service import pipeline_service

router = APIRouter()

class DiseaseDetectionResult(BaseModel):
    disease_name: str
    confidence: float
    severity_percentage: float
    is_healthy: bool
    gradcam_overlay_url: Optional[str] = None

class PestDetectionResult(BaseModel):
    pest_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    is_unreliable: bool  # True for Mango Hopper & Stem Borer (<100 annotated samples)

class RecommendationItem(BaseModel):
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
    scan_id: str
    scan_type: str
    farm_id: Optional[str] = None
    tree_id: Optional[str] = None
    ohi_score: float
    health_status: str
    disease_results: List[DiseaseDetectionResult]
    pest_results: List[PestDetectionResult]
    recommendations: List[RecommendationItem]
    nutrient_input_was_defaulted: bool = True
    caveats: List[str]
    penalty_breakdown: Optional[Dict[str, Any]] = None
    penalties: Optional[Dict[str, float]] = None
    image_url: Optional[str] = None
    pest_image_url: Optional[str] = None

@router.post("/", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Mango leaf image (JPEG/PNG)"),
    pest_file: Optional[UploadFile] = File(None, description="Optional canopy / pest trap image"),
    scan_type: str = Form("dual", description="'disease', 'pest', or 'dual'"),
    farm_id: Optional[str] = Form(None),
    tree_id: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None, description="Orchard latitude for microclimate lookups"),
    longitude: Optional[float] = Form(None, description="Orchard longitude for microclimate lookups")
):
    """
    Analyzes uploaded mango orchard imagery using live AI models:
    - Disease Detection (EfficientNet-B3 via PyTorch/timm)
    - Explainability Heatmap (Hook-based Grad-CAM)
    - Lesion Surface Area % (HSV color-space segmentation)
    - Pest Detection & Unreliable Class Gating (YOLOv8)
    - Microclimate Integration (Open-Meteo)
    - Deterministic Orchard Health Index (OHI)
    - Verified ICAR/TNAU Treatment Recommendations
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        # Graceful check for filenames ending in image extensions
        filename = file.filename or ""
        valid_exts = (".jpg", ".jpeg", ".png", ".webp")
        if not filename.lower().endswith(valid_exts):
            raise HTTPException(status_code=400, detail="Uploaded leaf file must be an image (JPEG/PNG/WebP).")

    try:
        leaf_bytes = await file.read()
        pest_bytes = await pest_file.read() if pest_file is not None else None

        result = pipeline_service.analyze_scan(
            leaf_bytes=leaf_bytes,
            leaf_filename=file.filename or "leaf.jpg",
            pest_bytes=pest_bytes,
            pest_filename=pest_file.filename if pest_file else None,
            latitude=latitude,
            longitude=longitude,
            scan_type=scan_type,
            farm_id=farm_id,
            tree_id=tree_id
        )

        return AnalysisResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")

@router.get("/{scan_id}", response_model=AnalysisResponse)
async def get_scan_by_id(scan_id: str):
    """Retrieves cached diagnosis and OHI results by scan ID."""
    scan = pipeline_service.get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail=f"Scan ID {scan_id} not found.")
    return AnalysisResponse(**scan)
