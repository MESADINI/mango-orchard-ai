import httpx
from typing import Dict, Any, Optional
from app.core.config import settings

class WeatherService:
    @staticmethod
    async def fetch_weather(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """Fetches 7-day weather forecast from Open-Meteo API."""
        url = f"{settings.OPEN_METEO_BASE_URL}/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "relative_humidity_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max"
            ],
            "timezone": "auto"
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e:
            print(f"WeatherService error: {e}")
        return None

weather_service = WeatherService()
