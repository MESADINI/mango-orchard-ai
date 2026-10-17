"""
AI Mango Orchard Health & Pest Intelligence — Deterministic Orchard Health Index (OHI)

Rules & Design Principles:
1. OHI is deterministic, not a black box. All scoring is rule-based Python with configurable weights.
2. All inputs, intermediate penalties, and final scores are fully inspectable.
3. A component that was not assessed (no trained model, uncertain prediction, no live weather, no nutrient
   sensor) is NEVER scored as healthy. It is excluded, the remaining weights are renormalised to sum to 1, and
   the component is listed in `missing_components` (with `is_partial: True`). Without at least one image-based
   component (disease or pest) there is no OHI: `ohi_score` is None and the status is "Unknown".
4. Pest detections flagged `is_unreliable` (Mango Hopper / Stem Borer, < 100 training samples) or
   `host_verified: False` (class trained on non-mango images) are shown to the user but never enter the OHI.

The formulas are documented in docs/OHI_METHODOLOGY.md, which must match this file.
"""

from typing import Any, Dict, Iterable, List, Optional

# Virulence multipliers per disease (alpha_c). Expert-assigned constants, not fitted to data; their influence is
# measured by the sensitivity analysis (ml/severity/ohi_sensitivity.py).
VIRULENCE_FACTORS = {
    "Healthy": 0.0,
    "Anthracnose": 1.00,       # High necrosis and fruit drop
    "Bacterial Canker": 0.95,  # Vascular systemic damage
    "Powdery Mildew": 0.85,    # Blossom and young flush destruction
    "Sooty Mold": 0.50         # Superficial saprophytic layer
}

# Pest damage impact factors per detection (beta_k). Expert-assigned constants, as above.
PEST_IMPACT_FACTORS = {
    "Fruit Fly": 18.0,
    "Mealy Bug": 12.0,
    "Scale Insect": 10.0,
    "Mango Hopper": 15.0,
    "Stem Borer": 35.0
}

COMPONENTS = ("disease", "pest", "weather", "nutrient")
IMAGE_COMPONENTS = ("disease", "pest")
DEFAULT_WEIGHTS = {"disease": 0.40, "pest": 0.35, "weather": 0.15, "nutrient": 0.10}

# A leaf classified Healthy whose colour segmentation finds at least this much lesion area is internally
# inconsistent: the OHI keeps the classifier's verdict (no disease penalty) but flags the scan for a retake or an
# expert check. Rule-of-thumb value; the HSV segmenter also marks shadows and dirt.
HEALTHY_LESION_INCONSISTENCY_PCT = 5.0

TIERS = (  # (lower bound, label, colour)
    (85.0, "Optimal Health", "#10B981"),
    (70.0, "Mild Stress / Warning", "#F59E0B"),
    (50.0, "High Infestation Risk", "#F97316"),
    (0.0, "Severe Outbreak / Critical", "#EF4444"),
)
UNKNOWN_STATUS = ("Unknown — insufficient data", "#6B7280")


def renormalise(weights: Dict[str, float], available: Iterable[str]) -> Dict[str, float]:
    """Weights of the available components rescaled to sum to 1; missing components get 0."""
    available = set(available)
    total = sum(w for c, w in weights.items() if c in available)
    return {c: (w / total if c in available and total > 0 else 0.0) for c, w in weights.items()}


def tier(score: Optional[float]):
    if score is None:
        return UNKNOWN_STATUS
    for lower, label, colour in TIERS:
        if score >= lower:
            return label, colour
    return TIERS[-1][1], TIERS[-1][2]


class OrchardHealthIndexEngine:
    def __init__(
        self,
        weight_disease: float = DEFAULT_WEIGHTS["disease"],
        weight_pest: float = DEFAULT_WEIGHTS["pest"],
        weight_weather: float = DEFAULT_WEIGHTS["weather"],
        weight_nutrient: float = DEFAULT_WEIGHTS["nutrient"],
        virulence: Optional[Dict[str, float]] = None,
        pest_impact: Optional[Dict[str, float]] = None,
    ):
        self.weights = {"disease": weight_disease, "pest": weight_pest, "weather": weight_weather,
                        "nutrient": weight_nutrient}
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError(f"weights must be non-negative with a positive sum: {self.weights}")
        self.virulence = dict(virulence or VIRULENCE_FACTORS)
        self.pest_impact = dict(pest_impact or PEST_IMPACT_FACTORS)

    def compute_disease_penalty(
        self,
        disease_name: str,
        confidence: float,
        severity_pct: float
    ) -> float:
        """
        Computes disease severity penalty S_d in range [0, 100].
        Formula: min(100, (alpha_c * 40) + (severity_pct * 0.6)) * confidence
        """
        name = disease_name.strip().title()
        if name not in self.virulence:
            raise ValueError(f"unknown disease class {disease_name!r}; expected one of {sorted(self.virulence)}")
        if name == "Healthy":
            return 0.0
        raw_penalty = (self.virulence[name] * 40.0) + (min(100.0, max(0.0, severity_pct)) * 0.60)
        return min(100.0, raw_penalty * max(0.0, min(1.0, confidence)))

    def compute_pest_penalty(
        self,
        detections: List[Dict[str, Any]]
    ) -> float:
        """
        Computes pest pressure penalty P_p in range [0, 100].
        Formula: min(100, sum(beta_k * confidence)). Callers pass verified detections only (see compute_ohi).
        """
        total_penalty = 0.0
        for det in detections:
            pest_name = det.get("pest_name", "")
            if pest_name not in self.pest_impact:
                raise ValueError(f"unknown pest class {pest_name!r}; expected one of {sorted(self.pest_impact)}")
            conf = float(det.get("confidence", 1.0))
            total_penalty += self.pest_impact[pest_name] * max(0.0, min(1.0, conf))
        return min(100.0, total_penalty)

    def compute_weather_vulnerability(
        self,
        relative_humidity: float,
        temperature: float,
        precipitation_mm: float
    ) -> float:
        """
        Computes weather vulnerability index W_v in range [0, 100].
        """
        # RH risk: linear from 0 at 50% RH to 100 at 100% RH
        rh_risk = max(0.0, min(100.0, (relative_humidity - 50.0) * 2.0))

        # Temperature risk: peak 100 at 27 C, -10 per degree, inside the 22-32 C window; 20 outside it
        if 22.0 <= temperature <= 32.0:
            temp_risk = 100.0 - (abs(temperature - 27.0) * 10.0)
        else:
            temp_risk = 20.0

        # Rain risk: leaf wetness promotes bacterial/fungal splash; saturates at 6.67 mm
        rain_risk = min(100.0, max(0.0, precipitation_mm) * 15.0)

        wv = (0.45 * rh_risk) + (0.35 * temp_risk) + (0.20 * rain_risk)
        return min(100.0, max(0.0, wv))

    def compute_ohi(
        self,
        disease_name: Optional[str] = "Healthy",
        disease_confidence: float = 1.0,
        disease_severity_pct: float = 0.0,
        pest_detections: Optional[List[Dict[str, Any]]] = None,
        relative_humidity: float = 65.0,
        temperature: float = 28.0,
        precipitation_mm: float = 0.0,
        nutrient_deficiency_score: Optional[float] = None,
        missing: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """
        Computes the deterministic OHI score [0, 100], or None when no image-based component was assessed.

        missing: components that were not assessed ("disease", "pest", "weather"); their inputs are ignored.
        disease_name=None also marks the disease component missing; nutrient is missing when
        nutrient_deficiency_score is None.
        """
        missing = set(missing)
        unknown = missing - set(COMPONENTS)
        if unknown:
            raise ValueError(f"unknown components in missing: {sorted(unknown)}")
        if disease_name is None:
            missing.add("disease")
        nutrient_input_was_defaulted = nutrient_deficiency_score is None
        if nutrient_input_was_defaulted:
            missing.add("nutrient")

        all_pests = list(pest_detections or [])
        gated = [d for d in all_pests if d.get("is_unreliable", False)]
        off_host = [d for d in all_pests if not d.get("is_unreliable", False) and d.get("host_verified", True) is False]
        verified_pests = [d for d in all_pests if d not in gated and d not in off_host]
        excluded_pests = len(gated) + len(off_host)

        penalties: Dict[str, Optional[float]] = {
            "disease": None if "disease" in missing else self.compute_disease_penalty(
                disease_name, disease_confidence, disease_severity_pct),
            "pest": None if "pest" in missing else self.compute_pest_penalty(verified_pests),
            "weather": None if "weather" in missing else self.compute_weather_vulnerability(
                relative_humidity, temperature, precipitation_mm),
            "nutrient": None if "nutrient" in missing else min(100.0, max(0.0, nutrient_deficiency_score)),
        }
        available = [c for c in COMPONENTS if c not in missing]
        weights = renormalise(self.weights, available)
        has_image_component = any(c in available for c in IMAGE_COMPONENTS)

        if has_image_component:
            total_penalty = sum(weights[c] * penalties[c] for c in available)
            ohi_score = round(max(0.0, min(100.0, 100.0 - total_penalty)), 2)
        else:
            total_penalty, ohi_score = None, None
        status, color = tier(ohi_score)

        caveats: List[str] = []
        if nutrient_input_was_defaulted:
            caveats.append("Nutrient deficiency is not measured (no sensor input); the nutrient component is "
                           "excluded and the other weights are renormalised.")
        if not has_image_component:
            caveats.append("No disease or pest assessment is available, so no Orchard Health Index is reported.")
        if gated:
            caveats.append(f"{len(gated)} detection(s) of unverified pest classes are shown but excluded from "
                           "the OHI (fewer than 100 training samples).")
        if off_host:
            caveats.append(f"{len(off_host)} pest detection(s) come from classes trained on non-mango images (traps, "
                           "other host plants); they are shown but excluded from the OHI.")
        inconsistent = (
            "disease" in available and disease_name.strip().title() == "Healthy"
            and disease_severity_pct >= HEALTHY_LESION_INCONSISTENCY_PCT
        )
        if inconsistent:
            caveats.append(f"The leaf was classified Healthy, but {disease_severity_pct:.1f}% of its area looks like "
                           "lesions. No disease penalty was applied; please retake the photo or ask an expert.")

        def r(v):
            return None if v is None else round(v, 2)

        breakdown = {
            "disease_severity_penalty": r(penalties["disease"]),
            "pest_pressure_penalty": r(penalties["pest"]),
            "weather_vulnerability_index": r(penalties["weather"]),
            "nutrient_penalty": r(penalties["nutrient"]),
            "weighted_total_penalty": r(total_penalty),
        }
        return {
            "ohi_score": ohi_score,
            "health_status": status,
            "badge_color": color,
            "penalty_breakdown": breakdown,
            "weights_used": {c: round(w, 4) for c, w in weights.items()},
            "nominal_weights": dict(self.weights),
            "missing_components": [c for c in COMPONENTS if c in missing and c != "nutrient"],
            "is_partial": any(c in missing for c in ("disease", "pest", "weather")),
            "nutrient_input_was_defaulted": nutrient_input_was_defaulted,
            "excluded_unverified_pest_detections": excluded_pests,
            "healthy_lesion_inconsistency": inconsistent,
            "caveats": caveats,
        }


ohi_engine = OrchardHealthIndexEngine()
