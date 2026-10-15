"""
AI Mango Orchard Health & Pest Intelligence — FastAPI Entry Point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.deps import get_store
from app.core.seed import seed_demo_data
from app.api import analyze, auth, chatbot, farm, report, weather
from app.services.pipeline_service import pipeline_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mango_orchard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_demo_data(get_store())
    logger.info("Loading ML pipeline...")
    status = pipeline_service.model_status()
    logger.info("ML pipeline ready: %s", status)
    if not all(status.values()):
        logger.warning("One or more models are not trained; affected scan components will be reported as unavailable.")
    yield
    logger.info("Shutting down Mango Orchard API.")


app = FastAPI(
    title="AI Mango Orchard Health & Pest Intelligence API",
    description="Backend API for mango leaf disease detection, hook-based Grad-CAM explainability, YOLOv8 pest "
                "detection, deterministic OHI scoring, and ICAR/TNAU-cited recommendations.",
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)

# Serves uploaded images, Grad-CAM heatmaps and pest overlays
settings.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(settings.MEDIA_DIR)), name="media")

app.include_router(analyze.router, prefix="/api/analyze", tags=["Analysis & OHI"])
app.include_router(weather.router, prefix="/api/weather", tags=["Weather & Forecasting"])
app.include_router(report.router, prefix="/api/report", tags=["Reports"])
app.include_router(chatbot.router, prefix="/api/chatbot", tags=["Bilingual Chatbot"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(farm.router, prefix="/api/farm", tags=["Farm Management"])


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "models": {
            "disease_detection": "EfficientNet-B3 (PyTorch / timm)",
            "explainability": "Hook-based Grad-CAM",
            "pest_detection": "YOLOv8 (Ultralytics)",
            **pipeline_service.model_status(),
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
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
