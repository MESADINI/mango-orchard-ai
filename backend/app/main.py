"""
AI Mango Orchard Health & Pest Intelligence — FastAPI Entry Point
Phase 5: Full production endpoint integration with live ML pipeline & static media serving.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api import analyze, auth, chatbot, farm, report, weather

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up ML pipeline and weights on startup
    print("🌿 Warming up Mango Orchard Health & Pest Intelligence pipeline...")
    try:
        from app.services.pipeline_service import pipeline_service
        pipeline_service.get_pipeline()
        print("✅ Live ML pipeline initialized successfully.")
    except Exception as e:
        print(f"⚠️ Note during pipeline warmup: {e}")
    yield
    print("🛑 Shutting down Mango Orchard API.")

app = FastAPI(
    title="AI Mango Orchard Health & Pest Intelligence API",
    description="Backend API for real-time mango leaf disease detection, hook-based Grad-CAM explainability, YOLOv8 pest intelligence, deterministic OHI scoring, and verified ICAR/TNAU recommendations.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount media directory for serving Grad-CAM heatmaps, pest detections, and generated reports
settings.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(settings.MEDIA_DIR)), name="media")

# API Routers
app.include_router(analyze.router, prefix="/api/analyze", tags=["Analysis & OHI"])
app.include_router(weather.router, prefix="/api/weather", tags=["Weather & Forecasting"])
app.include_router(report.router, prefix="/api/report", tags=["PDF Reports"])
app.include_router(chatbot.router, prefix="/api/chatbot", tags=["Bilingual Chatbot"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(farm.router, prefix="/api/farm", tags=["Farm Management"])

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": "1.0.0",
        "models": {
            "disease_detection": "EfficientNet-B3 (PyTorch / timm)",
            "explainability": "Hook-based Grad-CAM",
            "pest_detection": "YOLOv8 (Ultralytics)",
            "rules_enforced": [
                "Rule 1: Deterministic OHI",
                "Rule 2: ICAR/TNAU Citations",
                "Rule 3: Unreliable Class Flagging",
                "Rule 4: Disclosed Nutrient Default"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
