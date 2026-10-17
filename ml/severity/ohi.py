"""
AI Mango Orchard Health & Pest Intelligence — Deterministic Orchard Health Index (OHI)

Rules & Design Principles:
1. OHI is deterministic, not a black box. All scoring is rule-based Python with configurable weights.
2. All inputs, intermediate penalties, and final scores are fully inspectable.
3. Nutrient deficiency is a disclosed placeholder. N-input defaults to 0;
   response specifies nutrient_input_was_defaulted: True.
"""

from typing import Dict, Any, List, Optional

# Virulence Multipliers per pathogen (alpha_c)
VIRULENCE_FACTORS = {
    "Healthy": 0.0,
    "Anthracnose": 1.00,       # High necrosis and fruit drop
    "Bacterial Canker": 0.95,  # Vascular systemic damage
    "Powdery Mildew": 0.85,    # Blossom and young flush destruction
    "Sooty Mold": 0.50         # Superficial saprophytic layer
}

# Pest Damage Impact Factors per detection entity (beta_k)
PEST_IMPACT_FACTORS = {
    "Fruit Fly": 18.0,
    "Mealy Bug": 12.0,
    "Scale Insect": 10.0,
    "Mango Hopper": 15.0,
    "Stem Borer": 35.0
}

class OrchardHealthIndexEngine:
    def __init__(
        self,
        weight_disease: float = 0.40,
        weight_pest: float = 0.35,
        weight_weather: float = 0.15,
        weight_nutrient: float = 0.10
    ):
        self.w_d = weight_disease
        self.w_p = weight_pest
        self.w_w = weight_weather
        self.w_n = weight_nutrient

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
        if disease_name.strip().title() == "Healthy":
            return 0.0

        alpha = VIRULENCE_FACTORS.get(disease_name.strip().title(), 0.75)
        raw_penalty = (alpha * 40.0) + (min(100.0, severity_pct) * 0.60)
        return min(100.0, raw_penalty * max(0.0, min(1.0, confidence)))

    def compute_pest_penalty(
        self,
        detections: List[Dict[str, Any]]
    ) -> float:
        """
        Computes pest pressure penalty P_p in range [0, 100].
        Formula: min(100, sum(beta_k * confidence))
        """
        total_penalty = 0.0
        for det in detections:
            pest_name = det.get("pest_name", "")
            conf = float(det.get("confidence", 1.0))
            beta = PEST_IMPACT_FACTORS.get(pest_name, 10.0)
            total_penalty += beta * conf
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
        # RH risk: RH > 80% increases fungal spore germination
        rh_risk = max(0.0, min(100.0, (relative_humidity - 50.0) * 2.0)) if relative_humidity > 50 else 0.0

        # Temperature risk: 22C - 32C is optimal pathogen reproduction
        if 22.0 <= temperature <= 32.0:
            temp_risk = 100.0 - (abs(temperature - 27.0) * 10.0)
        else:
            temp_risk = 20.0

        # Rain risk: leaf wetness promotes bacterial/fungal splash
        rain_risk = min(100.0, precipitation_mm * 15.0)

        wv = (0.45 * rh_risk) + (0.35 * temp_risk) + (0.20 * rain_risk)
        return min(100.0, max(0.0, wv))

    def compute_ohi(
        self,
        disease_name: str = "Healthy",
        disease_confidence: float = 1.0,
        disease_severity_pct: float = 0.0,
        pest_detections: Optional[List[Dict[str, Any]]] = None,
        relative_humidity: float = 65.0,
        temperature: float = 28.0,
        precipitation_mm: float = 0.0,
        nutrient_deficiency_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes deterministic OHI score [0, 100].
        """
        if pest_detections is None:
            pest_detections = []

        # Policy: Disclosed nutrient deficiency placeholder
        nutrient_input_was_defaulted = False
        if nutrient_deficiency_score is None:
            nutrient_input_was_defaulted = True
            n_score = 0.0
            # Normalize weights without nutrient component
            w_d = self.w_d + 0.05
            w_p = self.w_p + 0.05
            w_w = self.w_w
            w_n = 0.0
        else:
            n_score = min(100.0, max(0.0, nutrient_deficiency_score))
            w_d, w_p, w_w, w_n = self.w_d, self.w_p, self.w_w, self.w_n

        s_d = self.compute_disease_penalty(disease_name, disease_confidence, disease_severity_pct)
        p_p = self.compute_pest_penalty(pest_detections)
        w_v = self.compute_weather_vulnerability(relative_humidity, temperature, precipitation_mm)

        total_penalty = (w_d * s_d) + (w_p * p_p) + (w_w * w_v) + (w_n * n_score)
        ohi_score = round(max(0.0, min(100.0, 100.0 - total_penalty)), 2)

        # Health Tiers
        if ohi_score >= 85.0:
            status = "Optimal Health"
            color = "#10B981"
        elif ohi_score >= 70.0:
            status = "Mild Stress / Warning"
            color = "#F59E0B"
        elif ohi_score >= 50.0:
            status = "High Infestation Risk"
            color = "#F97316"
        else:
            status = "Severe Outbreak / Critical"
            color = "#EF4444"

        return {
            "ohi_score": ohi_score,
            "health_status": status,
            "badge_color": color,
            "penalty_breakdown": {
                "disease_severity_penalty": round(s_d, 2),
                "pest_pressure_penalty": round(p_p, 2),
                "weather_vulnerability_index": round(w_v, 2),
                "nutrient_penalty": round(n_score, 2),
                "weighted_total_penalty": round(total_penalty, 2)
            },
            "penalties": {
                "disease_severity_penalty": round(s_d, 2),
                "pest_pressure_penalty": round(p_p, 2),
                "weather_vulnerability_index": round(w_v, 2),
                "nutrient_penalty": round(n_score, 2),
                "weighted_total_penalty": round(total_penalty, 2)
            },
            "weights_used": {
                "disease": w_d,
                "pest": w_p,
                "weather": w_w,
                "nutrient": w_n
            },
            "nutrient_input_was_defaulted": nutrient_input_was_defaulted,
            "caveats": [
                "Nutrient deficiency is a disclosed placeholder (sensor input was defaulted to 0)."
            ] if nutrient_input_was_defaulted else []
        }

ohi_engine = OrchardHealthIndexEngine()
