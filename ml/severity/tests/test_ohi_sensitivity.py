"""OHI sensitivity analysis: comparison metrics and the perturbation machinery, on synthetic scenarios."""

import numpy as np
import pytest

from ml.severity import ohi_sensitivity as S


def _scenarios(n=30, seed=0):
    rng = np.random.default_rng(seed)
    names = ["Healthy", "Anthracnose", "Bacterial Canker", "Powdery Mildew", "Sooty Mold"]
    out = []
    for i in range(n):
        out.append({"disease_name": names[i % 5], "disease_confidence": float(rng.uniform(0.85, 1.0)),
                    "low_confidence": i % 7 == 0, "severity_pct": float(rng.uniform(0, 40)),
                    "pests": [{"pest_name": "Fruit Fly", "confidence": 1.0, "is_unreliable": False}] * (i % 3),
                    "relative_humidity": float(rng.uniform(40, 100)), "temperature": float(rng.uniform(18, 36)),
                    "precipitation_mm": float(rng.uniform(0, 3))})
    return out


def test_compare_identity_and_shift():
    a = np.array([90.0, 80.0, 60.0, 40.0, np.nan])
    same = S.compare(a, a.copy())
    assert same["n"] == 4 and same["mean_abs_delta"] == 0 and same["tier_change_fraction"] == 0
    assert same["spearman_rho"] == pytest.approx(1.0)
    shifted = S.compare(a, a - 6.0)                        # 90->84 and 40->34 cross no tier; 80->74 neither
    assert shifted["mean_abs_delta"] == pytest.approx(6.0) and shifted["spearman_rho"] == pytest.approx(1.0)
    assert shifted["tier_change_fraction"] == pytest.approx(0.25)   # only 90 -> 84 leaves "Optimal Health"


def test_low_confidence_and_ablation_follow_the_app_rules():
    sc = _scenarios()
    nominal = S.score_all(sc, S.engine_with())
    assert not np.isnan(nominal).any()                     # pest + weather remain when disease is excluded
    no_pest_no_disease = S.score_all([dict(sc[0], pests=[])], S.engine_with(), drop="pest")
    assert np.isnan(no_pest_no_disease[0])                 # scenario 0 is low-confidence: no image component left


def test_analyse_reports_every_perturbation():
    res, nominal = S.analyse(_scenarios(), n_mc=20, seed=1)
    assert len(nominal) == 30
    assert {"weight:disease", "weight:pest", "weight:weather"} == {k.split("x")[0] for k in res["one_at_a_time"]
                                                                  if k.startswith("weight:")}
    assert set(res["ablation"]) == {"without disease", "without pest", "without weather"}
    for v in res["monte_carlo"].values():
        assert v["n_draws"] == 20 and -1 <= v["spearman_rho"]["median"] <= 1
    # scaling a weight by 1.0 would change nothing; scaling by 1.2 must change something
    assert res["one_at_a_time"]["weight:weather x1.2".replace(" ", "")]["mean_abs_delta"] > 0
