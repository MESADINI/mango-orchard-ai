import numpy as np
import pytest

from ml.disease_detection.metrics import (
    bootstrap_ci, classification_metrics, expected_calibration_error, fit_temperature, holm_adjust,
    mcnemar_exact, nll, ood_metrics, softmax, wilcoxon_paired,
)


def _onehot(pred, c=3, p=0.9):
    probs = np.full((len(pred), c), (1 - p) / (c - 1))
    probs[np.arange(len(pred)), pred] = p
    return probs


def test_classification_metrics_hand_computed():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    m = classification_metrics(y, _onehot(pred), ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["a"]["precision"] == pytest.approx(0.5) and m["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3) and m["per_class"]["b"]["recall"] == 1.0
    assert m["per_class"]["c"]["f1"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx(np.mean([0.5, 0.8, 2 / 3]))
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_classification_metrics_skip_absent_class_in_macro_averages():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, _onehot(np.array([0, 0, 1, 1])), ["a", "b", "c"])
    assert m["macro_f1"] == 1.0 and m["balanced_accuracy"] == 1.0
    assert m["per_class"]["c"]["recall"] is None and m["classes_evaluated"] == ["a", "b"]


def test_ece_zero_when_confidence_matches_accuracy():
    y = np.array([0] * 8 + [1] * 2)
    probs = np.tile([0.8, 0.2], (10, 1))      # always predicts 0 with confidence 0.8; accuracy 0.8
    assert expected_calibration_error(y, probs) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(y, np.tile([1.0, 0.0], (10, 1))) == pytest.approx(0.2)


def test_temperature_scaling_reduces_nll_of_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    logits = rng.normal(size=(500, 3))
    logits[np.arange(500), y] += 1.0
    logits *= 6.0                              # overconfident
    t = fit_temperature(logits, y)
    assert t > 1.5
    assert nll(logits, y, t) < nll(logits, y, 1.0)
    assert np.allclose(softmax(logits, t).sum(1), 1.0)


def test_mcnemar_counts_and_p_value():
    a = np.array([1] * 10 + [0] * 0 + [1] * 5, bool)
    b = np.array([0] * 10 + [1] * 5, bool)
    r = mcnemar_exact(a, b)
    assert r["a_right_b_wrong"] == 10 and r["a_wrong_b_right"] == 0
    assert r["p_value"] == pytest.approx(2 / 2 ** 10)
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_wilcoxon_reports_minimum_attainable_p_for_five_folds():
    r = wilcoxon_paired([0.9, 0.91, 0.92, 0.93, 0.94], [0.8, 0.81, 0.82, 0.83, 0.84])
    assert r["min_attainable_p"] == pytest.approx(0.0625)
    assert r["p_value"] == pytest.approx(0.0625)


def test_holm_adjust():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_ood_metrics_separable_scores():
    r = ood_metrics(np.linspace(0.9, 1.0, 100), np.linspace(0.1, 0.5, 50), threshold=0.6)
    assert r["auroc"] == 1.0 and r["fpr_at_95_tpr"] == 0.0
    assert r["in_accepted"] == 1.0 and r["out_rejected"] == 1.0


def test_bootstrap_ci_resamples_groups():
    y = np.array([0, 0, 1, 1])
    probs = _onehot(np.array([0, 0, 1, 0]), c=2)
    acc = lambda yy, pp: float((pp.argmax(1) == yy).mean())
    lo, hi = bootstrap_ci(y, probs, acc, groups=np.array([0, 0, 1, 1]), n_boot=200)
    assert 0.0 <= lo <= 0.75 <= hi <= 1.0


def test_temperature_is_flagged_when_val_is_perfectly_separated():
    from ml.disease_detection.metrics import fit_temperature, temperature_identifiable
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    separated = np.eye(5)[y] * 5.0 + rng.normal(scale=0.1, size=(200, 5))   # every sample correct
    T = fit_temperature(separated, y)
    assert T < 0.5 and not temperature_identifiable(separated, y, T)        # the fit only sharpens
    noisy = rng.normal(size=(200, 5)) + np.eye(5)[y] * 1.5                  # overlapping classes
    T_noisy = fit_temperature(noisy, y)
    assert 0.2 < T_noisy < 5 and temperature_identifiable(noisy, y, T_noisy)
