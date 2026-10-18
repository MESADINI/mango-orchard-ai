import logging
import sys
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
INSECURE_DEFAULT_JWT_SECRET = "change-me-in-production"


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Mango Orchard Health & Pest Intelligence"
    VERSION: str = "1.1.0"

    # Supabase (optional). When left at the placeholder values the backend uses
    # its built-in SQLite store for accounts and never contacts Supabase.
    SUPABASE_URL: str = "https://your-project.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY: str = "placeholder-service-role-key"
    SUPABASE_ANON_KEY: str = "placeholder-anon-key"

    # JWT security
    JWT_SECRET: str = INSECURE_DEFAULT_JWT_SECRET
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Demo account seeded into an empty database (set DEMO_USER_EMAIL empty to disable)
    DEMO_USER_EMAIL: str = "grower@mango.ai"
    DEMO_USER_PASSWORD: str = "orchard123"

    # CORS: comma-separated list of allowed browser origins
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Weather
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1"

    # Uploads
    MAX_UPLOAD_MB: int = 15
    PEST_CONFIDENCE_THRESHOLD: float = 0.25

    # Storage / ML paths
    BACKEND_DIR: Path = BACKEND_DIR
    REPO_DIR: Path = BACKEND_DIR.parent
    MEDIA_DIR: Path = BACKEND_DIR / "media"
    DATABASE_PATH: Path = BACKEND_DIR / "data" / "mango_orchard.db"
    DISEASE_WEIGHTS_PATH: Path = BACKEND_DIR.parent / "ml" / "disease_detection" / "weights" / "disease_classifier.pth"
    PEST_WEIGHTS_PATH: Path = BACKEND_DIR.parent / "ml" / "pest_detection" / "weights" / "pest_detector.pt"

    # Default orchard location (Krishnagiri mango cluster, Tamil Nadu)
    DEFAULT_LATITUDE: float = 12.5266
    DEFAULT_LONGITUDE: float = 78.2147

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def supabase_configured(self) -> bool:
        return "your-project" not in self.SUPABASE_URL and not self.SUPABASE_ANON_KEY.startswith("placeholder")

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()

# The ML package lives at <repo>/ml; make it importable as `ml.*`.
if str(settings.REPO_DIR) not in sys.path:
    sys.path.insert(0, str(settings.REPO_DIR))

if settings.JWT_SECRET.startswith(INSECURE_DEFAULT_JWT_SECRET):
    logger.warning("JWT_SECRET is set to the insecure default. Set a long random JWT_SECRET in backend/.env.")
