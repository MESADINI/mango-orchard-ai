"""Grad-CAM faithfulness metrics: AUC, perturbation schedule, pointing game and the summary statistics."""

import numpy as np
import pytest

from ml.explainability import faithfulness as F


def test_auc_trapezoid():
    assert F.auc([1, 1, 1]) == pytest.approx(1.0)
    assert F.auc([1, 0]) == pytest.approx(0.5)
    assert F.auc(np.linspace(1, 0, 21)) == pytest.approx(0.5)


def test_perturbation_batch_endpoints_and_order():
    img = np.ones((4, 4, 3), np.float32)
    base = np.zeros((4, 4, 3), np.float32)
    sal = np.arange(16, dtype=float).reshape(4, 4)               # most salient pixel = bottom-right
    order = F.ranking(sal)
    dele = F.perturbation_batch(img, base, order, 4, "deletion")
    ins = F.perturbation_batch(img, base, order, 4, "insertion")
    assert dele.shape == (5, 4, 4, 3)
    assert np.array_equal(dele[0], img) and np.array_equal(dele[-1], base)
    assert np.array_equal(ins[0], base) and np.array_equal(ins[-1], img)
    assert dele[1][3, 3, 0] == 0 and dele[1][0, 0, 0] == 1       # the salient quarter goes first
    assert dele[1].sum() == pytest.approx(img.sum() * 0.75)


def test_pointing_game():
    heat = np.zeros((50, 50))
    heat[10, 10] = 1
    lesion = np.zeros((50, 50), bool)
    lesion[10, 30] = True
    assert F.pointing_hit(heat, lesion, tolerance=15) is False
    assert F.pointing_hit(heat, lesion, tolerance=20) is True
    assert F.pointing_hit(heat, np.zeros_like(lesion)) is None


def test_centre_prior_peaks_in_the_middle():
    p = F.centre_prior(9)
    assert np.unravel_index(np.argmax(p), p.shape) == (4, 4)


def test_summary_compares_with_baselines_and_gates_pointing():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(30):
        r = {}
        for m, d, i in (("gradcam", 0.2, 0.8), ("random", 0.5, 0.5), ("centre", 0.4, 0.6)):
            r[f"{m}_deletion_auc"] = d + rng.normal(scale=0.02)
            r[f"{m}_insertion_auc"] = i + rng.normal(scale=0.02)
        rows.append(r)
    s = F.summarise({"per_image": rows, "pointing": [{"hit": True}] * 3, "steps": 20})
    assert s["deletion"]["gradcam_vs_random"]["fraction_gradcam_better"] == 1.0
    assert s["insertion"]["gradcam_vs_centre"]["wilcoxon_p"] < 0.001
    assert "not run" in s["pointing_game"]["status"]
