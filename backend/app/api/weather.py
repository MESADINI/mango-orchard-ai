"""
Weather & disease-spread forecast endpoint.
Uses the same Open-Meteo fetcher and epidemiological forecaster as the scan pipeline.
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
    temperature_max: float
    temperature_min: float
    relative_humidity_mean: float
    precipitation_sum: float
    anthracnose_spread_risk: float  # percent
    powdery_mildew_spread_risk: float  # percent
    disease_spread_risk: str  # "Low", "Moderate", "High"
    dominant_threat: str


class WeatherRiskResponse(BaseModel):
    latitude: float
    longitude: float
    current_temperature: Optional[float] = None
    current_humidity: Optional[float] = None
    current_precipitation: Optional[float] = None
    forecast_days: List[DailyForecast]
    pathogen_alerts: List[str]


def _dominant_threat(day: dict) -> str:
    if day["alert_level"] == "Low":
        return "Unfavorable conditions for rapid foliar infection"
    if day["anthracnose_spread_risk"] >= day["powdery_mildew_spread_risk"]:
        return "Anthracnose spore germination favored by humidity and rain"
    return "Powdery Mildew favored by dry days and cool nights"


@router.get("/forecast", response_model=WeatherRiskResponse)
async def get_weather_forecast(
    latitude: float = Query(settings.DEFAULT_LATITUDE, ge=-90, le=90, description="Latitude of mango orchard"),
    longitude: float = Query(settings.DEFAULT_LONGITUDE, ge=-180, le=180, description="Longitude of mango orchard"),
):
    """
    Fetches the 7-day microclimate forecast from Open-Meteo (no key required) and computes
    daily Anthracnose and Powdery Mildew spread risks. Returns 503 if Open-Meteo is unreachable.
    """
    data = await run_in_threadpool(fetch_open_meteo_forecast, latitude, longitude)
    if not data or "daily" not in data:
        raise HTTPException(status_code=503, detail="Weather service (Open-Meteo) is currently unreachable.")

    current = data.get("current") or {}
    days = []
    for day in forecast_disease_spread(data["daily"]):
        days.append(DailyForecast(
            date=day["date"],
            temperature_max=day["temperature_max"],
            temperature_min=day["temperature_min"],
            relative_humidity_mean=day["relative_humidity"],
            precipitation_sum=day["precipitation_mm"],
            anthracnose_spread_risk=day["anthracnose_spread_risk"],
            powdery_mildew_spread_risk=day["powdery_mildew_spread_risk"],
            disease_spread_risk=day["alert_level"],
            dominant_threat=_dominant_threat(day),
        ))

    alerts = []
    if any(d.disease_spread_risk == "High" for d in days[:3]):
        alerts.append("High disease pressure expected within the next 72 hours. Inspect flush and flower panicles.")

    return WeatherRiskResponse(
        latitude=latitude,
        longitude=longitude,
        current_temperature=current.get("temperature_2m"),
        current_humidity=current.get("relative_humidity_2m"),
        current_precipitation=current.get("precipitation"),
        forecast_days=days,
        pathogen_alerts=alerts,
    )
