"""
Weather & disease-favourable-conditions forecast endpoint.
Uses the same Open-Meteo fetcher and rule set (ml/forecasting/forecaster.py) as the scan pipeline. The levels say
whether the weather matches published disease-favourable conditions; they are not validated outbreak probabilities.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.config import settings
from ml.forecasting.forecaster import forecast_disease_spread
from ml.forecasting.weather import fetch_open_meteo_forecast

router = APIRouter()


class DailyForecast(BaseModel):
    date: str
    temperature_max: Optional[float] = None
    temperature_min: Optional[float] = None
    relative_humidity_mean: Optional[float] = None
    precipitation_sum: Optional[float] = None
    hours_rh95_t25_30: Optional[int] = None
    anthracnose_risk: str          # "High" | "Moderate" | "Low" | "Unknown"
    powdery_mildew_risk: str
    bacterial_canker_risk: str
    disease_spread_risk: str       # highest of the three
    dominant_threat: str


class WeatherRiskResponse(BaseModel):
    latitude: float
    longitude: float
    current_temperature: Optional[float] = None
    current_humidity: Optional[float] = None
    current_precipitation: Optional[float] = None
    forecast_days: List[DailyForecast]
    pathogen_alerts: List[str]


THREAT_TEXT = {
    "anthracnose_risk": "Anthracnose: long humid spell (RH >= 95% at 25-30 C)",
    "powdery_mildew_risk": "Powdery mildew: cool nights (10-13 C), warm days (27-31 C), RH 82-91%",
    "bacterial_canker_risk": "Bacterial canker: RH >= 90% at 25-30 C",
}
LEVELS = ("Unknown", "Low", "Moderate", "High")


def _dominant_threat(day: dict) -> str:
    if day["alert_level"] in ("Low", "Unknown"):
        return ("Weather does not match the published disease-favourable conditions" if day["alert_level"] == "Low"
                else "Not enough weather data for this day")
    key = max(THREAT_TEXT, key=lambda k: LEVELS.index(day[k]))
    return THREAT_TEXT[key]


@router.get("/forecast", response_model=WeatherRiskResponse)
async def get_weather_forecast(
    latitude: float = Query(settings.DEFAULT_LATITUDE, ge=-90, le=90, description="Latitude of mango orchard"),
    longitude: float = Query(settings.DEFAULT_LONGITUDE, ge=-180, le=180, description="Longitude of mango orchard"),
):
    """
    Fetches the 7-day microclimate forecast from Open-Meteo (no key required) and computes
    daily levels for anthracnose, powdery mildew and bacterial canker weather. Returns 503 if Open-Meteo is unreachable.
    """
    data = await run_in_threadpool(fetch_open_meteo_forecast, latitude, longitude)
    if not data or "daily" not in data:
        raise HTTPException(status_code=503, detail="Weather service (Open-Meteo) is currently unreachable.")

    current = data.get("current") or {}
    days = []
    for day in forecast_disease_spread(data["daily"], data.get("hourly")):
        days.append(DailyForecast(
            date=day["date"],
            temperature_max=day["temperature_max"],
            temperature_min=day["temperature_min"],
            relative_humidity_mean=day["relative_humidity"],
            precipitation_sum=day["precipitation_mm"],
            hours_rh95_t25_30=day["hours_rh95_t25_30"],
            anthracnose_risk=day["anthracnose_risk"],
            powdery_mildew_risk=day["powdery_mildew_risk"],
            bacterial_canker_risk=day["bacterial_canker_risk"],
            disease_spread_risk=day["alert_level"],
            dominant_threat=_dominant_threat(day),
        ))

    alerts = []
    if any(d.disease_spread_risk == "High" for d in days[:3]):
        alerts.append("Weather in the next 72 hours matches published disease-favourable conditions. If trees are "
                      "flowering or flushing, inspect panicles and young leaves.")

    return WeatherRiskResponse(
        latitude=latitude,
        longitude=longitude,
        current_temperature=current.get("temperature_2m"),
        current_humidity=current.get("relative_humidity_2m"),
        current_precipitation=current.get("precipitation"),
        forecast_days=days,
        pathogen_alerts=alerts,
    )
