"""
AI Mango Orchard Health & Pest Intelligence — Disease Spread Forecaster
Combines microclimate forecast with pathogen growth epidemiology curves.
"""

from typing import Dict, Any, List

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
        t_max = max_t[i] if i < len(max_t) else 30.0
        t_min = min_t[i] if i < len(min_t) else 22.0
        rh = rh_mean[i] if i < len(rh_mean) else 65.0
        p = precip[i] if i < len(precip) else 0.0

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
