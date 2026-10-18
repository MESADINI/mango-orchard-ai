"""Lesion segmentation on synthetic leaves with known lesion area."""

import cv2
import numpy as np
import pytest

from ml.severity.severity import compute_lesion_severity

LEAF_GREEN = (60, 140, 50)      # RGB, hue ~ 50 (OpenCV scale), clearly green
SOOTY_BLACK = (25, 25, 25)      # low saturation, very dark: the sooty mould coating
BROWN = (120, 60, 20)           # necrotic lesion


def _leaf(bg=(235, 235, 235), size=400):
    img = np.full((size, size, 3), bg, np.uint8)
    leaf = np.zeros((size, size), np.uint8)
    cv2.ellipse(leaf, (size // 2, size // 2), (150, 70), 0, 0, 360, 1, -1)
    img[leaf > 0] = LEAF_GREEN
    return img, leaf > 0


def _disc(shape, centre, r):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, centre, r, 1, -1)
    return m > 0


def test_healthy_leaf_has_no_lesions():
    img, _ = _leaf()
    assert compute_lesion_severity(img)["severity_percentage"] == 0.0


def test_sooty_black_coating_is_counted_as_lesion():
    img, leaf = _leaf()
    patch = _disc(leaf.shape, (170, 200), 35) & leaf
    img[patch] = SOOTY_BLACK
    res = compute_lesion_severity(img, return_masks=True)
    expected = 100 * patch.sum() / leaf.sum()
    assert res["severity_percentage"] == pytest.approx(expected, abs=1.5)
    assert res["leaf_mask"].sum() == pytest.approx(leaf.sum(), rel=0.03)   # the black patch is part of the leaf


def test_large_coating_on_the_margin_stays_in_the_leaf():
    img, leaf = _leaf()
    half = leaf.copy()
    half[:, 200:] = False                     # the whole left half of the leaf is black
    img[half] = SOOTY_BLACK
    res = compute_lesion_severity(img)
    assert res["severity_percentage"] == pytest.approx(100 * half.sum() / leaf.sum(), abs=3)


def test_brown_lesions_are_counted():
    img, leaf = _leaf()
    spots = (_disc(leaf.shape, (150, 190), 15) | _disc(leaf.shape, (260, 210), 20)) & leaf
    img[spots] = BROWN
    expected = 100 * spots.sum() / leaf.sum()
    assert compute_lesion_severity(img)["severity_percentage"] == pytest.approx(expected, abs=1.0)


def test_background_soil_and_shadow_are_not_lesions():
    img, leaf = _leaf()
    img[:60, :] = BROWN                                      # soil strip along the top, away from the leaf
    img[340:, 50:350] = (40, 40, 40)                         # dark shadow below the leaf, not touching it
    res = compute_lesion_severity(img)
    assert res["severity_percentage"] == 0.0
    assert res["total_leaf_pixels"] == pytest.approx(leaf.sum(), rel=0.03)


def test_dark_background_does_not_swallow_the_image():
    img, leaf = _leaf(bg=(20, 20, 20))                       # leaf photographed on black cloth
    res = compute_lesion_severity(img)
    assert res["total_leaf_pixels"] < 1.1 * leaf.sum()
    assert res["severity_percentage"] < 3.0


def test_no_leaf_returns_zero():
    res = compute_lesion_severity(np.full((100, 100, 3), 240, np.uint8))
    assert res == {"severity_percentage": 0.0, "lesion_pixels": 0, "total_leaf_pixels": 0}


def test_large_images_are_processed_at_bounded_resolution():
    img, _ = _leaf(size=400)
    big = cv2.resize(img, (2000, 2000), interpolation=cv2.INTER_NEAREST)
    res = compute_lesion_severity(big)
    assert res["total_leaf_pixels"] <= 512 * 512
