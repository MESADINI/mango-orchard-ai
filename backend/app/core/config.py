from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Mango Orchard Health & Pest Intelligence"
    API_V1_STR: str = "/api"
    
    # Supabase credentials
    SUPABASE_URL: str = "https://your-project.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY: str = "placeholder-service-role-key"
    SUPABASE_ANON_KEY: str = "placeholder-anon-key"
    
    # JWT security
    JWT_SECRET: str = "change-me-in-production-secure-key-32chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Weather
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1"
    
    # Storage / ML paths
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
    REPO_DIR: Path = BACKEND_DIR.parent
    MEDIA_DIR: Path = BACKEND_DIR / "media"
    DISEASE_WEIGHTS_PATH: Path = REPO_DIR / "ml" / "disease_detection" / "weights" / "efficientnet_b3_best.pth"
    PEST_WEIGHTS_PATH: Path = REPO_DIR / "ml" / "pest_detection" / "weights" / "yolov8_best.pt"
    
    # Default orchard location (Krishnagiri mango cluster, Tamil Nadu)
    DEFAULT_LATITUDE: float = 12.5266
    DEFAULT_LONGITUDE: float = 78.2147
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
settings.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
(settings.MEDIA_DIR / "gradcam").mkdir(parents=True, exist_ok=True)
(settings.MEDIA_DIR / "pests").mkdir(parents=True, exist_ok=True)
(settings.MEDIA_DIR / "reports").mkdir(parents=True, exist_ok=True)
(settings.MEDIA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
