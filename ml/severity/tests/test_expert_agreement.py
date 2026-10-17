"""Expert-rating scaffolding: blind template, Spearman evaluation and its minimum-size guard."""

import csv

import pytest

from ml.severity import expert_agreement as E


def _scenarios(path, n=60):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_label", "predicted", "confidence", "low_confidence", "severity_pct", "n_pests",
                    "weather_time", "rh", "temp_c", "precip_mm", "ohi_nominal"])
        for i in range(n):
            w.writerow([f"img{i}.jpg", "Healthy", "Healthy", "0.99", "False", "1.0", 0, "t", 60, 27, 0,
                        "" if i == 0 else f"{40 + i:.2f}"])
    return path


def test_template_is_blind_and_stratified(tmp_path):
    sc = _scenarios(tmp_path / "scenarios.csv")
    out = E.make_template(20, scenarios=sc, out=tmp_path / "t.csv")
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 20 and "ohi_nominal" not in rows[0] and "true_label" not in rows[0]
    assert len({r["scenario_id"] for r in rows}) == 20
    assert "0" not in {r["scenario_id"] for r in rows}       # scenario without an OHI is never offered
    tiers = {E.tier(40 + int(r["scenario_id"]))[0] for r in rows}
    assert len(tiers) == 4


def _filled(path, rater, ids, fn):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=E.FIELDS)
        w.writeheader()
        for s in ids:
            w.writerow({"scenario_id": s, "rater_id": rater, "rating_0_100": fn(int(s))})
    return path


def test_evaluate_spearman_and_guard(tmp_path):
    sc = _scenarios(tmp_path / "scenarios.csv")
    ids = [str(i) for i in range(1, 31)]
    a = _filled(tmp_path / "a.csv", "A", ids, lambda i: 40 + i)            # perfectly monotone with the OHI
    b = _filled(tmp_path / "b.csv", "B", ids, lambda i: 100 - (i % 7))     # unrelated
    res = E.evaluate([a], scenarios=sc, out_dir=tmp_path)
    assert res["spearman_ohi_vs_mean_rating"] == pytest.approx(1.0) and res["inter_rater_mean_spearman"] is None
    res2 = E.evaluate([a, b], scenarios=sc, out_dir=tmp_path)
    assert res2["n_raters"] == 2 and res2["inter_rater_mean_spearman"] is not None
    few = _filled(tmp_path / "few.csv", "C", ids[:5], lambda i: 50)
    with pytest.raises(SystemExit):
        E.evaluate([few], scenarios=sc, out_dir=tmp_path)
    bad = _filled(tmp_path / "bad.csv", "D", ids, lambda i: 150)
    with pytest.raises(ValueError):
        E.evaluate([bad], scenarios=sc, out_dir=tmp_path)
