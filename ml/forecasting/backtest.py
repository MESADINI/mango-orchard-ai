#!/usr/bin/env python3
"""
Back-test of the disease-weather rules on 10 years of hourly reanalysis weather for Krishnagiri (Tamil Nadu).

    python ml/forecasting/backtest.py [--start 2015-01-01 --end 2024-12-31] [--source auto|open-meteo|nasa-power]

Data: the Open-Meteo historical archive. If it is unreachable (--source auto), NASA POWER hourly (MERRA-2) is used
instead, through the same rules; the source actually used is recorded in backtest.json.

There are no outbreak records for this location, so this is NOT an accuracy test of disease prediction. It answers:
  1. Agreement: how often did the PREVIOUS heuristic rules (copied verbatim below from the pre-Phase-5 forecaster)
     raise a High alert on the days that match the conditions the NHM IPM bulletin documents as favourable?
     (precision / recall / specificity / Cohen's kappa against the documented conditions)
  2. Climatology: on how many days per month does Krishnagiri weather meet each documented condition?
  3. Interpretation sensitivity: how much do those counts change under reasonable alternative readings of the text?
Outputs: results/forecaster/{backtest.json, monthly.md, daily_features.csv, monthly.png}
Raw responses are cached in ml/datasets/raw/weather/ (git-ignored); their SHA-256 hashes are recorded.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.forecasting import forecaster as fc  # noqa: E402
from ml.forecasting.weather import fetch_nasa_power_hourly, fetch_open_meteo_archive  # noqa: E402
from ml.utils.experiment import git_info, write_json  # noqa: E402

LATITUDE, LONGITUDE = 12.5266, 78.2147          # Krishnagiri, same default as ml/pipeline.py
CACHE_DIR = REPO_ROOT / "ml" / "datasets" / "raw" / "weather"
OUT = REPO_ROOT / "results" / "forecaster"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ----------------------------------------------------------------------------- previous rules (verbatim logic)
def legacy_rules(t_max, t_min, rh, p):
    """The forecaster's rules before Phase 5 (git 72abe1a, ml/forecasting/forecaster.py), unchanged."""
    anthracnose_risk = min(1.0, (rh / 100.0) * (1.0 if p > 1.5 else 0.4))
    diurnal_diff = t_max - t_min
    mildew_risk = 0.85 if (diurnal_diff > 10.0 and 50.0 <= rh <= 70.0) else 0.25
    alert = "High" if anthracnose_risk > 0.7 or mildew_risk > 0.7 else "Moderate" if anthracnose_risk > 0.4 else "Low"
    return {"anthracnose_high": anthracnose_risk > 0.7, "powdery_mildew_high": mildew_risk > 0.7, "alert": alert}


# ----------------------------------------------------------------------------- metrics
def agreement(pred, ref):
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    tp, fp = int((pred & ref).sum()), int((pred & ~ref).sum())
    fn, tn = int((~pred & ref).sum()), int((~pred & ~ref).sum())
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    div = lambda a, b: a / b if b else None  # noqa: E731
    return {"n_days": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "reference_positive_days": tp + fn, "predicted_positive_days": tp + fp,
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "cohen_kappa": div(po - pe, 1 - pe)}


def monthly_mean_days(dates, flags, n_years):
    counts = defaultdict(int)
    for d, f in zip(dates, flags):
        if f:
            counts[int(d[5:7])] += 1
    return {MONTHS[m - 1]: counts[m] / n_years for m in range(1, 13)}


# ----------------------------------------------------------------------------- main
def load_weather(start: str, end: str, source: str = "auto"):
    """(payload, source description, attempts). payload has an Open-Meteo-shaped "hourly" block."""
    attempts = []
    if source in ("auto", "open-meteo"):
        payload = fetch_open_meteo_archive(LATITUDE, LONGITUDE, start, end,
                                           cache_path=CACHE_DIR / f"krishnagiri_{start}_{end}_hourly.json")
        attempts.append({"source": "open-meteo", "ok": payload is not None})
        if payload is not None:
            payload["_source"] = "Open-Meteo historical weather archive (reanalysis), hourly"
            return payload, attempts
    if source in ("auto", "nasa-power"):
        payload = fetch_nasa_power_hourly(LATITUDE, LONGITUDE, start, end, cache_dir=CACHE_DIR)
        attempts.append({"source": "nasa-power", "ok": payload is not None})
        if payload is not None:
            return payload, attempts
    return None, attempts


def run(start: str, end: str, payload=None, source: str = "auto"):
    attempts = []
    if payload is None:
        payload, attempts = load_weather(start, end, source)
        if payload is None:
            print(f"No weather source reachable ({attempts}) and no cache; nothing to report.")
            return None
    feats = [f for f in fc.daily_features(None, payload["hourly"]) if start <= f["date"] <= end]
    complete = [f for f in feats if None not in (f["temperature_max"], f["temperature_min"],
                                                f["relative_humidity"], f["precipitation_mm"],
                                                f["hours_rh95_t25_30"])]
    dates = [f["date"] for f in complete]
    n_years = len({d[:4] for d in dates})
    documented = [fc.assess_day(f)["risks"] for f in complete]
    legacy = [legacy_rules(f["temperature_max"], f["temperature_min"], f["relative_humidity"], f["precipitation_mm"])
              for f in complete]

    ref = {d: [r[d] == "High" for r in documented] for d in fc.DISEASES}
    hourly = payload["hourly"]
    hours = [(rh, tc) for ts, rh, tc in zip(hourly["time"], hourly["relative_humidity_2m"], hourly["temperature_2m"])
             if start <= ts[:10] <= end and rh is not None and tc is not None]
    rh_summary = {
        "hourly_rh_max": max(rh for rh, _ in hours),
        "hourly_rh_p99": float(sorted(rh for rh, _ in hours)[int(0.99 * (len(hours) - 1))]),
        "hours_rh_ge_95_per_year": sum(rh >= 95 for rh, _ in hours) / n_years,
        "hours_rh_ge_90_per_year": sum(rh >= 90 for rh, _ in hours) / n_years,
        "hours_rh_ge_90_t25_30_per_year": sum(rh >= 90 and 25 <= tc <= 30 for rh, tc in hours) / n_years,
    }
    co_occurrence_caveat = (
        [f"That anthracnose-favourable weather never occurs at Krishnagiri. In this reanalysis, hourly RH >= 95% is "
         f"common ({rh_summary['hours_rh_ge_95_per_year']:.0f} h/year) but never at 25-30 C, and RH >= 90% at "
         f"25-30 C occurs {rh_summary['hours_rh_ge_90_t25_30_per_year']:.1f} h/year, so the hour-by-hour reading of "
         "the bulletin's condition (RH >= 95% while 25-30 C) is never met. Whether the bulletin means simultaneous "
         "hours is not stated; canopy leaf wetness is not measured; a gridded reanalysis smooths local extremes."]
        if not any(ref["anthracnose"]) else [])
    result = {
        "location": {"name": "Krishnagiri, Tamil Nadu", "latitude": LATITUDE, "longitude": LONGITUDE,
                     "grid_elevation_m": payload.get("elevation"), "timezone": payload.get("timezone")},
        "data": {"source": payload.get("_source", "unknown"), "source_attempts": attempts,
                 "request_url": payload.get("_request_url") or payload.get("_request_urls"),
                 "sha256": payload.get("_sha256"),
                 "period": [start, end], "days_total": len(feats), "days_complete": len(complete),
                 "years": n_years},
        "agreement_previous_rules_vs_documented": {
            "anthracnose": agreement([l["anthracnose_high"] for l in legacy], ref["anthracnose"]),
            "powdery_mildew": agreement([l["powdery_mildew_high"] for l in legacy], ref["powdery_mildew"]),
            "any_high_alert_vs_any_documented": agreement(
                [l["alert"] == "High" for l in legacy],
                [any(ref[d][i] for d in fc.DISEASES) for i in range(len(complete))]),
        },
        "documented_conditions_days_per_year": {d: sum(ref[d]) / n_years for d in fc.DISEASES},
        "documented_conditions_moderate_or_high_days_per_year": {
            d: sum(r[d] in ("High", "Moderate") for r in documented) / n_years for d in fc.DISEASES},
        "previous_rules_high_days_per_year": {
            "anthracnose": sum(l["anthracnose_high"] for l in legacy) / n_years,
            "powdery_mildew": sum(l["powdery_mildew_high"] for l in legacy) / n_years,
            "any": sum(l["alert"] == "High" for l in legacy) / n_years},
        "monthly_mean_days": {
            **{f"documented_{d}": monthly_mean_days(dates, ref[d], n_years) for d in fc.DISEASES},
            "previous_anthracnose_high": monthly_mean_days(dates, [l["anthracnose_high"] for l in legacy], n_years),
            "previous_powdery_mildew_high": monthly_mean_days(dates, [l["powdery_mildew_high"] for l in legacy],
                                                              n_years)},
        "interpretation_sensitivity_days_per_year": interpretation_sensitivity(complete, n_years),
        "weather_summary": {
            "hours_rh95_t25_30_max_in_a_day": max(f["hours_rh95_t25_30"] for f in complete),
            "days_with_any_hour_rh95_t25_30_per_year":
                sum(f["hours_rh95_t25_30"] > 0 for f in complete) / n_years,
            "temperature_min_lowest": min(f["temperature_min"] for f in complete),
            "days_tmin_10_13_per_year": sum(10 <= f["temperature_min"] <= 13 for f in complete) / n_years,
            "days_mean_rh_82_91_per_year": sum(82 <= f["relative_humidity"] <= 91 for f in complete) / n_years,
            **rh_summary},
        "can_claim": [
            "How often Krishnagiri reanalysis weather met the conditions the NHM IPM bulletin calls favourable, by month.",
            "How well the previous heuristic rules agreed with those documented conditions.",
        ],
        "cannot_claim": [
            "That any alert predicts an outbreak, or any detection rate, lead time or false-alarm rate for real disease: "
            "no outbreak or disease-incidence records were available for validation.",
            "That the powdery-mildew thresholds hold in Tamil Nadu: the bulletin states them for the UP plains.",
            "Orchard microclimate: reanalysis is a coarse grid estimate (ERA5 ~25 km; MERRA-2 0.5 x 0.625 deg), "
            "not a station or canopy measurement.",
            "Crop-stage relevance: favourable weather matters only when flowers, young flush or fruit are present.",
            *co_occurrence_caveat,
        ],
        "thresholds_source": "https://agritech.tnau.ac.in/horticulture/pdf/tech_bulletin/national/IPM-Mango-Revised-Sept2011.pdf",
        "git": git_info(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "backtest.json", result)
    _write_daily_csv(complete, documented, legacy)
    _write_monthly(result)
    return result


def interpretation_sensitivity(complete, n_years):
    """Days/year under alternative readings of the bulletin text (our main reading is marked 'main')."""
    def per_year(cond):
        return sum(cond(f) for f in complete) / n_years
    return {
        "anthracnose": {
            "main: >=12 h RH>=95 at 25-30 C": per_year(lambda f: f["hours_rh95_t25_30"] >= 12),
            ">=6 h RH>=95 at 25-30 C": per_year(lambda f: f["hours_rh95_t25_30"] >= 6),
            ">=1 h RH>=95 at 25-30 C": per_year(lambda f: f["hours_rh95_t25_30"] >= 1),
        },
        "powdery_mildew": {
            "main: Tmin 10-13, Tmax 27-31, mean RH 82-91": per_year(
                lambda f: 10 <= f["temperature_min"] <= 13 and 27 <= f["temperature_max"] <= 31
                and 82 <= f["relative_humidity"] <= 91),
            "Tmin 10-13 and Tmax 27-31 (RH ignored)": per_year(
                lambda f: 10 <= f["temperature_min"] <= 13 and 27 <= f["temperature_max"] <= 31),
            "thresholds widened by 2 C / 5% RH": per_year(
                lambda f: 8 <= f["temperature_min"] <= 15 and 25 <= f["temperature_max"] <= 33
                and 77 <= f["relative_humidity"] <= 96),
        },
        "bacterial_canker": {
            "main: mean RH>=90 and mean T 25-30": per_year(
                lambda f: f["relative_humidity"] >= 90 and 25 <= f["temperature_mean"] <= 30),
            "mean RH>=85 and mean T 25-30": per_year(
                lambda f: f["relative_humidity"] >= 85 and 25 <= f["temperature_mean"] <= 30),
        },
    }


def _write_daily_csv(complete, documented, legacy):
    keys = ["date", "temperature_min", "temperature_max", "temperature_mean", "relative_humidity",
            "precipitation_mm", "hours_rh95_t25_30"]
    with open(OUT / "daily_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys + [f"documented_{d}" for d in fc.DISEASES] + ["previous_anthracnose_high",
                                                                      "previous_powdery_mildew_high"])
        for feat, doc, leg in zip(complete, documented, legacy):
            w.writerow([round(feat[k], 3) if isinstance(feat[k], float) else feat[k] for k in keys]
                       + [doc[d] for d in fc.DISEASES] + [int(leg["anthracnose_high"]), int(leg["powdery_mildew_high"])])


def _write_monthly(result):
    m = result["monthly_mean_days"]
    rows = ["| Month | Anthracnose (documented) | Powdery mildew (documented) | Bacterial canker (documented) | "
            "Anthracnose (previous rule) | Powdery mildew (previous rule) |", "|---|---|---|---|---|---|"]
    for mon in MONTHS:
        rows.append(f"| {mon} | {m['documented_anthracnose'][mon]:.1f} | {m['documented_powdery_mildew'][mon]:.1f} | "
                    f"{m['documented_bacterial_canker'][mon]:.1f} | {m['previous_anthracnose_high'][mon]:.1f} | "
                    f"{m['previous_powdery_mildew_high'][mon]:.1f} |")
    d = result["data"]
    (OUT / "monthly.md").write_text(
        f"Mean days per month meeting each condition ('High'), Krishnagiri, {d['period'][0]} to {d['period'][1]} "
        f"({d['days_complete']} complete days; source: {d['source']}). 'Documented' = NHM IPM bulletin conditions; "
        "'previous rule' = the pre-Phase-5 heuristic. Weather only: no disease records.\n\n" + "\n".join(rows) + "\n",
        encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(12)
    for i, (key, label) in enumerate([("documented_anthracnose", "Anthracnose (documented)"),
                                      ("documented_powdery_mildew", "Powdery mildew (documented)"),
                                      ("documented_bacterial_canker", "Bacterial canker (documented)"),
                                      ("previous_anthracnose_high", "Anthracnose (previous rule)")]):
        ax.bar(x + (i - 1.5) * 0.2, [m[key][mon] for mon in MONTHS], 0.2, label=label)
    ax.set_xticks(x, MONTHS)
    ax.set_ylabel("mean days / month")
    ax.set_title("Days meeting disease-favourable weather conditions, Krishnagiri")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "monthly.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--source", default="auto", choices=["auto", "open-meteo", "nasa-power"])
    a = ap.parse_args()
    r = run(a.start, a.end, source=a.source)
    if r:
        import json
        print(json.dumps({k: r[k] for k in ("data", "agreement_previous_rules_vs_documented",
                                            "documented_conditions_days_per_year", "weather_summary",
                                            "interpretation_sensitivity_days_per_year")}, indent=1))
