"""Disease-favourable weather rules: thresholds from the NHM IPM bulletin, unknown when data is missing."""

from ml.forecasting.forecaster import assess_day, daily_features, forecast_disease_spread


def _hourly(date, temps, rhs, rain=None):
    return {"time": [f"{date}T{h:02d}:00" for h in range(len(temps))], "temperature_2m": temps,
            "relative_humidity_2m": rhs, "precipitation": rain or [0.0] * len(temps)}


def test_anthracnose_needs_12_humid_hours_in_the_25_30_window():
    def level(n_humid, temp=27.0):
        h = _hourly("2026-01-01", [temp] * n_humid + [31.0] * (24 - n_humid), [96.0] * n_humid + [60.0] * (24 - n_humid))
        return forecast_disease_spread(None, h)[0]["anthracnose_risk"]
    assert level(12) == "High"
    assert level(8) == "Moderate"
    assert level(3) == "Low"
    assert level(14, temp=23.0) == "Low"          # humid but too cool


def test_powdery_mildew_thresholds():
    base = {"temperature_min": 12.0, "temperature_max": 29.0, "relative_humidity": 85.0,
            "temperature_mean": 20.5, "hours_rh95_t25_30": 0, "precipitation_mm": 0.0}
    assert assess_day(base)["risks"]["powdery_mildew"] == "High"
    assert assess_day({**base, "temperature_min": 18.0})["risks"]["powdery_mildew"] == "Moderate"
    assert assess_day({**base, "temperature_min": 18.0, "relative_humidity": 60.0})["risks"]["powdery_mildew"] == "Low"


def test_bacterial_canker_thresholds():
    day = {"temperature_min": 24.0, "temperature_max": 30.0, "temperature_mean": 27.0, "relative_humidity": 92.0,
           "hours_rh95_t25_30": 0, "precipitation_mm": 4.0}
    assert assess_day(day)["risks"]["bacterial_canker"] == "High"
    assert assess_day({**day, "relative_humidity": 70.0})["risks"]["bacterial_canker"] == "Moderate"
    assert assess_day(day)["rain_splash_day"] is True


def test_missing_data_is_unknown_not_defaulted():
    out = forecast_disease_spread({"time": ["2026-01-01"], "temperature_2m_max": [None], "temperature_2m_min": [20.0],
                                   "relative_humidity_2m_mean": [80.0], "precipitation_sum": [None]})
    d = out[0]
    assert d["temperature_max"] is None and d["precipitation_mm"] is None
    assert d["anthracnose_risk"] == "Unknown"          # daily data alone cannot count humid hours
    assert d["powdery_mildew_risk"] == "Unknown" and d["rain_splash_day"] is None


def test_hourly_features_and_incomplete_days():
    h = _hourly("2026-01-01", [20.0 + i % 10 for i in range(24)], [80.0] * 24, [0.1] * 24)
    f = daily_features(None, h)[0]
    assert (f["temperature_min"], f["temperature_max"]) == (20.0, 29.0)
    assert round(f["precipitation_mm"], 6) == 2.4
    short = daily_features(None, _hourly("2026-01-02", [25.0] * 10, [96.0] * 10))[0]
    assert short["hours_rh95_t25_30"] is None and short["temperature_max"] is None


def test_alert_level_is_the_highest_known_risk():
    day = {"temperature_min": 12.0, "temperature_max": 29.0, "relative_humidity": 85.0, "temperature_mean": 20.5,
           "hours_rh95_t25_30": None, "precipitation_mm": 0.0}
    a = assess_day(day)
    assert a["risks"]["anthracnose"] == "Unknown" and a["alert_level"] == "High"
