"""
AI Mango Orchard Health & Pest Intelligence — Open-Meteo Weather Fetcher
Fetches current conditions and a 7-day forecast (daily + hourly) without requiring API keys, and the historical
archive used by the forecaster back-test.
"""

import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

OPEN_METEO_BASE_URL = os.environ.get("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1").rstrip("/")
OPEN_METEO_ARCHIVE_URL = os.environ.get("OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1").rstrip("/")
# hourly variables the disease rules need (ml/forecasting/forecaster.py), identical for forecast and archive
HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation"


def _get_json(url: str, timeout: float) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MangoOrchardAI/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
            logger.warning("Open-Meteo returned HTTP %s", response.status)
    except Exception as e:
        logger.warning("Weather fetch error: %s", e)
    return None


def fetch_open_meteo_forecast(latitude: float, longitude: float, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Fetches the 7-day weather forecast from Open-Meteo. Returns None if the service is unreachable."""
    params = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum",
        "hourly": HOURLY_VARS,
        "timezone": "auto",
        "forecast_days": 7,
    })
    return _get_json(f"{OPEN_METEO_BASE_URL}/forecast?{params}", timeout)


def archive_url(latitude: float, longitude: float, start_date: str, end_date: str,
                timezone: str = "Asia/Kolkata") -> str:
    params = urllib.parse.urlencode({
        "latitude": latitude, "longitude": longitude, "start_date": start_date, "end_date": end_date,
        "hourly": HOURLY_VARS, "timezone": timezone,
    })
    return f"{OPEN_METEO_ARCHIVE_URL}/archive?{params}"


NASA_POWER_URL = os.environ.get("NASA_POWER_URL", "https://power.larc.nasa.gov/api/temporal/hourly/point")


def fetch_nasa_power_hourly(latitude: float, longitude: float, start_date: str, end_date: str,
                            cache_dir: Optional[Path] = None, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
    """
    Hourly NASA POWER data (MERRA-2 reanalysis; T2M, RH2M, PRECTOTCORR; local solar time), converted to the
    Open-Meteo "hourly" shape so the same rules apply. Fallback when Open-Meteo is unreachable. Fetched one calendar
    quarter per request (whole years time out), retried with back-off, cached per quarter. Fill values (-999) -> None.
    Returns {"hourly": ..., "_request_urls": [...], "_sha256": {chunk: sha}, "_source": "NASA POWER"} or None.
    """
    hourly = {"time": [], "temperature_2m": [], "relative_humidity_2m": [], "precipitation": []}
    urls, shas = [], {}
    quarters = [(f"{y}-{m:02d}-01", f"{y}-{m + 2:02d}-{(31 if m + 2 in (3, 12) else 30)}")
                for y in range(int(start_date[:4]), int(end_date[:4]) + 1) for m in (1, 4, 7, 10)]
    for q_start, q_end in quarters:
        if q_end < start_date or q_start > end_date:
            continue
        s = max(start_date, q_start).replace("-", "")
        e = min(end_date, q_end).replace("-", "")
        url = NASA_POWER_URL + "?" + urllib.parse.urlencode({
            "parameters": "T2M,RH2M,PRECTOTCORR", "community": "AG", "longitude": longitude, "latitude": latitude,
            "start": s, "end": e, "format": "JSON", "time-standard": "LST"})
        cache = Path(cache_dir) / f"nasa_power_{latitude}_{longitude}_{s}_{e}.json" if cache_dir else None
        if cache is not None and cache.exists():
            raw = cache.read_bytes()
        else:
            data = None
            for attempt in range(4):              # the service throttles bursts; back off and retry
                data = _get_json(url, timeout)
                if data is not None:
                    break
                time.sleep(15 * (attempt + 1))
            if data is None:
                return None
            raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(raw)
        urls.append(url)
        shas[f"{s}-{e}"] = hashlib.sha256(raw).hexdigest()
        params = json.loads(raw)["properties"]["parameter"]
        fill = -999.0
        for key in sorted(params["T2M"]):
            hourly["time"].append(f"{key[:4]}-{key[4:6]}-{key[6:8]}T{key[8:10]}:00")
            for src, dst in (("T2M", "temperature_2m"), ("RH2M", "relative_humidity_2m"),
                             ("PRECTOTCORR", "precipitation")):
                v = params[src].get(key)
                hourly[dst].append(None if v is None or v == fill else float(v))
    return {"hourly": hourly, "_request_urls": urls, "_sha256": shas, "_source": "NASA POWER (MERRA-2), hourly, LST"}


def fetch_open_meteo_archive(latitude: float, longitude: float, start_date: str, end_date: str,
                             cache_path: Optional[Path] = None, timeout: float = 120.0) -> Optional[Dict[str, Any]]:
    """
    Hourly historical weather (Open-Meteo archive: reanalysis, not station observations). With cache_path, the raw
    response is stored on first download and re-used afterwards, so a back-test is reproducible offline.
    Returns the payload plus "_request_url" and "_sha256" (of the cached bytes), or None if unreachable.
    """
    url = archive_url(latitude, longitude, start_date, end_date)
    if cache_path is not None and Path(cache_path).exists():
        raw = Path(cache_path).read_bytes()
    else:
        data = _get_json(url, timeout)
        if data is None:
            return None
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_bytes(raw)
    payload = json.loads(raw)
    payload["_request_url"] = url
    payload["_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload
