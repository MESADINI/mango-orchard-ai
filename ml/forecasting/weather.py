"""
AI Mango Orchard Health & Pest Intelligence — Open-Meteo Weather Fetcher
Fetches current conditions and a 7-day daily forecast without requiring API keys.
"""

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

OPEN_METEO_BASE_URL = os.environ.get("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1").rstrip("/")


def fetch_open_meteo_forecast(latitude: float, longitude: float, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Fetches the 7-day weather forecast from Open-Meteo. Returns None if the service is unreachable."""
    params = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "auto",
        "forecast_days": 7,
    })
    url = f"{OPEN_METEO_BASE_URL}/forecast?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MangoOrchardAI/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
            logger.warning("Open-Meteo returned HTTP %s", response.status)
    except Exception as e:
        logger.warning("Weather fetch error: %s", e)
    return None
