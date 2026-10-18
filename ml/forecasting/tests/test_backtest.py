"""Forecaster back-test: agreement metrics, monthly aggregation, the frozen legacy rules and an offline run."""

import json

import pytest

from ml.forecasting import backtest as B


def test_agreement_counts_and_kappa():
    a = B.agreement([1, 1, 0, 0], [1, 0, 1, 0])
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (1, 1, 1, 1)
    assert a["precision"] == 0.5 and a["recall"] == 0.5 and a["cohen_kappa"] == pytest.approx(0.0)
    perfect = B.agreement([1, 0, 1], [1, 0, 1])
    assert perfect["cohen_kappa"] == pytest.approx(1.0)
    none_ref = B.agreement([1, 0], [0, 0])
    assert none_ref["recall"] is None and none_ref["precision"] == 0.0


def test_monthly_mean_days():
    m = B.monthly_mean_days(["2020-01-01", "2020-01-02", "2021-01-05", "2021-07-01"], [1, 1, 1, 0], 2)
    assert m["Jan"] == 1.5 and m["Jul"] == 0 and len(m) == 12


def test_legacy_rules_are_frozen():
    """Verbatim pre-Phase-5 logic: high anthracnose needs rain > 1.5 mm and RH > 70%."""
    assert B.legacy_rules(30, 20, 80, 2.0)["anthracnose_high"] is True
    assert B.legacy_rules(30, 20, 80, 1.0)["anthracnose_high"] is False
    assert B.legacy_rules(32, 20, 60, 0.0)["powdery_mildew_high"] is True    # diurnal range > 10, RH 50-70
    assert B.legacy_rules(28, 20, 60, 0.0)["alert"] == "Low"


def _payload(days=2):
    times, t, rh, p = [], [], [], []
    for d in range(1, days + 1):
        for h in range(24):
            times.append(f"2020-03-{d:02d}T{h:02d}:00")
            humid = d == 1 and h < 13                     # day 1: 13 h at RH 96%, 26 C -> anthracnose High
            t.append(26.0 if humid else 31.0)
            rh.append(96.0 if humid else 60.0)
            p.append(0.0)
    return {"hourly": {"time": times, "temperature_2m": t, "relative_humidity_2m": rh, "precipitation": p},
            "_source": "synthetic"}


def test_offline_run_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT", tmp_path)
    res = B.run("2020-03-01", "2020-03-02", payload=_payload())
    assert res["data"]["days_complete"] == 2
    assert res["documented_conditions_days_per_year"]["anthracnose"] == 1.0
    assert res["weather_summary"]["hours_rh_ge_95_per_year"] == 13
    assert not any("never occurs" in c for c in res["cannot_claim"])   # the caveat appears only when it applies
    assert json.loads((tmp_path / "backtest.json").read_text(encoding="utf-8"))["data"]["source"] == "synthetic"
    assert (tmp_path / "monthly.md").exists()
