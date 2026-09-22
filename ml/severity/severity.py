"""
AI Mango Orchard Health & Pest Intelligence — Lesion Severity Analysis
Segment diseased foliar tissue surface area fraction using color space analysis & thresholding.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any

def compute_lesion_severity(image_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Computes the percentage of leaf surface area afflicted by necrosis/lesions.
    Returns:
        severity_percentage: float [0.0, 100.0]
        mask: binary mask of detected lesions
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Segment healthy green leaf area
    # Hue: 25 to 85 represents vegetative green/yellow-green
    lower_green = np.array([25, 35, 35])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Segment necrotic lesions (brown, black, dark grey, reddish spots)
    lower_lesion = np.array([0, 40, 20])
    upper_lesion = np.array([22, 255, 180])
    lesion_mask = cv2.inRange(hsv, lower_lesion, upper_lesion)

    total_leaf_pixels = np.count_nonzero(green_mask) + np.count_nonzero(lesion_mask)

    if total_leaf_pixels == 0:
        return {
            "severity_percentage": 0.0,
            "lesion_pixels": 0,
            "total_leaf_pixels": 0
        }

    lesion_pixels = np.count_nonzero(lesion_mask)
    severity_pct = (lesion_pixels / total_leaf_pixels) * 100.0

    return {
        "severity_percentage": round(min(100.0, severity_pct), 2),
        "lesion_pixels": lesion_pixels,
        "total_leaf_pixels": total_leaf_pixels
    }
