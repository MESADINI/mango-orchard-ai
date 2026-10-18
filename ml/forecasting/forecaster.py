"""
AI Mango Orchard Health & Pest Intelligence — Disease-Favourable Weather Rules

Flags days whose weather matches the conditions that published mango guidance describes as favourable for each
disease. It is NOT a validated outbreak predictor: no outbreak records were available to test it against
(see ml/forecasting/backtest.py and docs/phases/PHASE_5.md for what it can and cannot claim).

Source of every threshold: National Horticulture Mission, "IPM Schedule for Mango Pests", Extension Bulletin No. 1
(2012), https://agritech.tnau.ac.in/horticulture/pdf/tech_bulletin/national/IPM-Mango-Revised-Sept2011.pdf
  Anthracnose:      "Temperature 25-30 [C], 95% Relative humidity for 12 hours is conducive for development"
  Powdery mildew:   "Minimum temperature (10-13 C), maximum 27-31 C and RH 82-91% are most conducive"
                    (stated for the UP plains; transfer to Tamil Nadu is an assumption)
  Bacterial canker: "favoured by high humidity (90% RH), moderate temperatures (25-30 C)"

Our interpretation of the text (documented, not from the source):
  Anthracnose      High: >= 12 hours in the day with RH >= 95% and 25 <= T <= 30 C; Moderate: 6-11 such hours.
  Powdery mildew   High: all three of Tmin in [10, 13], Tmax in [27, 31], daily mean RH in [82, 91]; Moderate: two.
  Bacterial canker High: daily mean RH >= 90% and daily mean T in [25, 30]; Moderate: one of the two.
  Missing inputs -> "Unknown" (never a default value). Anthracnose needs hourly data.
Risk only matters when susceptible tissue is present (flowers, young flush, fruit); the app does not know the
orchard's crop stage.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

LEVELS = ("Unknown", "Low", "Moderate", "High")
DISEASES = ("anthracnose", "powdery_mildew", "bacterial_canker")

ANTHRACNOSE_RH, ANTHRACNOSE_T, ANTHRACNOSE_HOURS = 95.0, (25.0, 30.0), 12
POWDERY_TMIN, POWDERY_TMAX, POWDERY_RH = (10.0, 13.0), (27.0, 31.0), (82.0, 91.0)
CANKER_RH, CANKER_T = 90.0, (25.0, 30.0)
MIN_HOURS_PER_DAY = 20          # a day with fewer valid hourly values is treated as missing for hourly features


def _num(v) -> Optional[float]:
    return float(v) if v is not None else None


def _at(values: List[Any], i: int) -> Optional[float]:
    return _num(values[i]) if i < len(values) else None


def _within(x: Optional[float], lo_hi) -> Optional[bool]:
    return None if x is None else lo_hi[0] <= x <= lo_hi[1]


def daily_features(daily: Optional[Dict[str, Any]] = None, hourly: Optional[Dict[str, Any]] = None
                   ) -> List[Dict[str, Any]]:
    """
    Per-day features. Hourly data (Open-Meteo "hourly": time, temperature_2m, relative_humidity_2m, precipitation)
    is preferred and gives every feature; daily data alone gives everything except the anthracnose hour count.
    """
    days: Dict[str, Dict[str, Any]] = {}
    if hourly and hourly.get("time"):
        acc = defaultdict(lambda: {"t": [], "rh": [], "p": [], "both": 0})
        t, rh, p = hourly.get("temperature_2m", []), hourly.get("relative_humidity_2m", []), \
            hourly.get("precipitation", [])
        for i, stamp in enumerate(hourly["time"]):
            a = acc[stamp[:10]]
            ti, rhi, pi = _at(t, i), _at(rh, i), _at(p, i)
            if ti is not None and rhi is not None:
                a["t"].append(ti)
                a["rh"].append(rhi)
                if rhi >= ANTHRACNOSE_RH and ANTHRACNOSE_T[0] <= ti <= ANTHRACNOSE_T[1]:
                    a["both"] += 1
            if pi is not None:
                a["p"].append(pi)
        for date, a in acc.items():
            ok = len(a["t"]) >= MIN_HOURS_PER_DAY
            days[date] = {
                "date": date,
                "temperature_max": max(a["t"]) if ok else None,
                "temperature_min": min(a["t"]) if ok else None,
                "temperature_mean": sum(a["t"]) / len(a["t"]) if ok else None,
                "relative_humidity": sum(a["rh"]) / len(a["rh"]) if ok else None,
                "precipitation_mm": sum(a["p"]) if len(a["p"]) >= MIN_HOURS_PER_DAY else None,
                "hours_rh95_t25_30": a["both"] if ok else None,
            }
    if daily and daily.get("time"):
        for i, date in enumerate(daily["time"]):
            d = days.setdefault(date, {"date": date, "hours_rh95_t25_30": None})
            tmax, tmin = _at(daily.get("temperature_2m_max", []), i), _at(daily.get("temperature_2m_min", []), i)
            for key, value in (("temperature_max", tmax), ("temperature_min", tmin),
                               ("relative_humidity", _at(daily.get("relative_humidity_2m_mean", []), i)),
                               ("precipitation_mm", _at(daily.get("precipitation_sum", []), i))):
                if d.get(key) is None:
                    d[key] = value
            if d.get("temperature_mean") is None:
                d["temperature_mean"] = (tmax + tmin) / 2 if tmax is not None and tmin is not None else None
    return [days[k] for k in sorted(days)]


def _level_from_conditions(conds: List[Optional[bool]]) -> str:
    if any(c is None for c in conds):
        return "Unknown"
    met = sum(conds)
    return "High" if met == len(conds) else "Moderate" if met == len(conds) - 1 else "Low"


def assess_day(f: Dict[str, Any]) -> Dict[str, Any]:
    hours = f.get("hours_rh95_t25_30")
    anth = "Unknown" if hours is None else \
        "High" if hours >= ANTHRACNOSE_HOURS else "Moderate" if hours >= ANTHRACNOSE_HOURS / 2 else "Low"
    pm = _level_from_conditions([_within(f.get("temperature_min"), POWDERY_TMIN),
                                 _within(f.get("temperature_max"), POWDERY_TMAX),
                                 _within(f.get("relative_humidity"), POWDERY_RH)])
    rh = f.get("relative_humidity")
    canker = _level_from_conditions([None if rh is None else rh >= CANKER_RH,
                                     _within(f.get("temperature_mean"), CANKER_T)])
    risks = {"anthracnose": anth, "powdery_mildew": pm, "bacterial_canker": canker}
    known = [r for r in risks.values() if r != "Unknown"]
    alert = max(known, key=LEVELS.index) if known else "Unknown"
    return {"risks": risks, "alert_level": alert,
            "rain_splash_day": None if f.get("precipitation_mm") is None else f["precipitation_mm"] > 0}


def forecast_disease_spread(daily_weather: Optional[Dict[str, Any]], hourly_weather: Optional[Dict[str, Any]] = None
                            ) -> List[Dict[str, Any]]:
    """Per-day disease-favourable-weather levels for the forecast period."""
    out = []
    for f in daily_features(daily_weather, hourly_weather):
        a = assess_day(f)
        out.append({
            "date": f["date"],
            "temperature_max": f.get("temperature_max"),
            "temperature_min": f.get("temperature_min"),
            "relative_humidity": f.get("relative_humidity"),
            "precipitation_mm": f.get("precipitation_mm"),
            "hours_rh95_t25_30": f.get("hours_rh95_t25_30"),
            "anthracnose_risk": a["risks"]["anthracnose"],
            "powdery_mildew_risk": a["risks"]["powdery_mildew"],
            "bacterial_canker_risk": a["risks"]["bacterial_canker"],
            "rain_splash_day": a["rain_splash_day"],
            "alert_level": a["alert_level"],
        })
    return out
