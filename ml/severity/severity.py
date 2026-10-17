"""
AI Mango Orchard Health & Pest Intelligence — Lesion Severity Analysis
Segment diseased foliar tissue surface area fraction using color space analysis & thresholding.

Steps (all thresholds below are hand-set, not learned; validation against masks: ml/severity/validate.py):
1. Leaf region: green pixels, morphologically closed, largest connected component, holes filled. Lesion-coloured
   pixels connected to the leaf inside its convex hull are added (lesions on the margin). Everything else is
   background (paper, soil, shadow) and never counts as lesion.
2. Lesions inside the leaf region:
   - brown/red-brown necrosis (hue 0-22, saturation >= 40, value 20-200), and
   - dark tissue, e.g. the black, low-saturation coating of sooty mould: value below DARK_FRACTION x the median
     value of the leaf's green pixels (relative, so dark-green leaves are not marked) or below DARK_ABSOLUTE.
   Dark pixels count as lesion even when their hue is green.
3. Severity = lesion pixels / leaf-region pixels x 100.
Known limitation: white powdery mildew coating is not segmented as lesion (it looks like a specular highlight).
"""

from typing import Any, Dict

import cv2
import numpy as np

MAX_SIDE = 512                 # images are downscaled so kernel sizes mean the same thing for every photo
GREEN_LOW, GREEN_HIGH = np.array([25, 35, 35]), np.array([85, 255, 255])
BROWN_LOW, BROWN_HIGH = np.array([0, 40, 20]), np.array([22, 255, 200])
DARK_FRACTION = 0.55
DARK_ABSOLUTE = 45
BORDER_GROWTH_LIMIT = 0.10      # max extra share of the image border the leaf may gain by growing through dark tissue


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.astype(np.uint8) * 255
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:-1, 1:-1] = flood
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(canvas, ff_mask, (0, 0), 255)
    holes = canvas[1:-1, 1:-1] == 0
    return mask | holes


def _border_fraction(mask: np.ndarray) -> float:
    """Fraction of the image border pixels covered by the mask."""
    border = np.concatenate([mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]])
    return float(border.mean())


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def segment_leaf_and_lesions(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Boolean leaf-region and lesion masks at the (possibly downscaled) working resolution."""
    h, w = image_rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH) > 0
    brown = cv2.inRange(hsv, BROWN_LOW, BROWN_HIGH) > 0
    empty = np.zeros(green.shape, bool)
    if green.sum() < 0.002 * green.size:          # no leaf visible
        return {"leaf": empty, "lesion": empty}

    k = max(3, int(0.02 * max(green.shape)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    leaf = _largest_component(cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0)

    median_leaf_v = float(np.median(v[green & leaf])) if (green & leaf).any() else 128.0
    dark = v < max(DARK_ABSOLUTE, DARK_FRACTION * median_leaf_v)

    # Lesion-coloured tissue connected to the leaf belongs to it (a heavy sooty coating can cover half the leaf).
    # If growing through dark/brown pixels reaches much more of the image border than the green leaf does, the
    # background itself is dark (soil, cloth, shadow): then only tissue inside the leaf's convex hull is added.
    grown = _largest_component(cv2.morphologyEx((leaf | brown | dark).astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0)
    if _border_fraction(grown) > _border_fraction(leaf) + BORDER_GROWTH_LIMIT:
        hull = np.zeros(green.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(leaf.astype(np.uint8))), 1)
        grown = _largest_component(grown & ((hull > 0) | leaf))
    leaf = _fill_holes(grown)

    lesion = leaf & (dark | (brown & ~green))
    return {"leaf": leaf, "lesion": lesion}


def compute_lesion_severity(image_rgb: np.ndarray, return_masks: bool = False) -> Dict[str, Any]:
    """
    Computes the percentage of leaf surface area afflicted by necrosis/lesions.
    Returns:
        severity_percentage: float [0.0, 100.0]
        lesion_pixels, total_leaf_pixels: pixel counts at the working resolution (longest side <= 512)
        leaf_mask, lesion_mask: boolean masks (only with return_masks=True)
    """
    masks = segment_leaf_and_lesions(image_rgb)
    total_leaf_pixels = int(masks["leaf"].sum())
    lesion_pixels = int(masks["lesion"].sum())
    severity_pct = 100.0 * lesion_pixels / total_leaf_pixels if total_leaf_pixels else 0.0
    result: Dict[str, Any] = {
        "severity_percentage": round(min(100.0, severity_pct), 2),
        "lesion_pixels": lesion_pixels,
        "total_leaf_pixels": total_leaf_pixels,
    }
    if return_masks:
        result.update(leaf_mask=masks["leaf"], lesion_mask=masks["lesion"])
    return result
