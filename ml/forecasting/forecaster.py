"""
AI Mango Orchard Health & Pest Intelligence — Disease Spread Forecaster
Combines microclimate forecast with pathogen growth epidemiology curves.
"""

from typing import Dict, Any, List


def _value_at(values: List[Any], i: int, default: float) -> float:
    """Returns values[i] as float, or default when missing or null (Open-Meteo may return null)."""
    if i < len(values) and values[i] is not None:
        return float(values[i])
    return default


def forecast_disease_spread(daily_weather: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Computes disease outbreak probability for the next 7 days based on
    humidity, precipitation, and thermal diurnal ranges.
    """
    dates = daily_weather.get("time", [])
    max_t = daily_weather.get("temperature_2m_max", [])
    min_t = daily_weather.get("temperature_2m_min", [])
    rh_mean = daily_weather.get("relative_humidity_2m_mean", [])
    precip = daily_weather.get("precipitation_sum", [])

    forecasts = []
    for i in range(len(dates)):
        t_max = _value_at(max_t, i, 30.0)
        t_min = _value_at(min_t, i, 22.0)
        rh = _value_at(rh_mean, i, 65.0)
        p = _value_at(precip, i, 0.0)

        # Anthracnose: Colletotrichum requires >80% RH and >2mm rain
        anthracnose_risk = min(1.0, (rh / 100.0) * (1.0 if p > 1.5 else 0.4))
        
        # Powdery mildew: Favored by dry days, cool nights (diurnal gap > 10C) and RH 50-70%
        diurnal_diff = t_max - t_min
        mildew_risk = 0.85 if (diurnal_diff > 10.0 and 50.0 <= rh <= 70.0) else 0.25

        forecasts.append({
            "date": dates[i],
            "temperature_max": t_max,
            "temperature_min": t_min,
            "relative_humidity": rh,
            "precipitation_mm": p,
            "anthracnose_spread_risk": round(anthracnose_risk * 100, 1),
            "powdery_mildew_spread_risk": round(mildew_risk * 100, 1),
            "alert_level": "High" if anthracnose_risk > 0.7 or mildew_risk > 0.7 else "Moderate" if anthracnose_risk > 0.4 else "Low"
        })

    return forecasts
