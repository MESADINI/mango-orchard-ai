"""
AI Mango Orchard Health & Pest Intelligence — Open-Meteo Weather Fetcher
Fetches hourly & daily forecast data without requiring API keys.
"""

import urllib.request
import json
from typing import Dict, Any, Optional

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_open_meteo_forecast(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """Fetches real-time 7-day weather forecast from Open-Meteo."""
    query = (
        f"{OPEN_METEO_URL}?latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        f"&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum"
        f"&timezone=auto"
    )
    try:
        req = urllib.request.Request(query, headers={"User-Agent": "MangoOrchardAI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data
    except Exception as e:
        print(f"Weather fetch error: {e}")
    return None
