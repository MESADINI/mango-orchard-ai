"""Shared test helpers (sample images and stubbed weather services)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_LEAF = REPO_ROOT / "ml" / "datasets" / "sample_leaves" / "anthracnose_sample.jpg"
SAMPLE_PEST = REPO_ROOT / "ml" / "datasets" / "sample_pests" / "fruit_fly_sample.jpg"


def fake_weather(latitude, longitude):
    """Deterministic Open-Meteo-shaped payload (includes a null, as the real API sometimes returns)."""
    return {
        "current": {"temperature_2m": 27.0, "relative_humidity_2m": 85.0, "precipitation": 3.0},
        "daily": {
            "time": [f"2026-10-{d:02d}" for d in range(15, 22)],
            "temperature_2m_max": [31.0, 30.5, 32.0, 29.0, 28.5, 30.0, None],
            "temperature_2m_min": [22.0, 21.5, 20.0, 18.0, 17.0, 19.0, 20.0],
            "relative_humidity_2m_mean": [88.0, 82.0, 60.0, 55.0, 65.0, 90.0, 70.0],
            "precipitation_sum": [5.0, 2.0, 0.0, 0.0, 0.0, 12.0, None],
        },
        # hourly data for the first day only: 14 h at RH 96% and 26 C (an anthracnose-favourable spell)
        "hourly": {
            "time": [f"2026-10-15T{h:02d}:00" for h in range(24)],
            "temperature_2m": [26.0] * 14 + [29.0] * 10,
            "relative_humidity_2m": [96.0] * 14 + [70.0] * 10,
            "precipitation": [0.5] * 6 + [0.0] * 18,
        },
    }


def offline_weather(latitude, longitude):
    return None
