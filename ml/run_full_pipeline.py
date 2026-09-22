#!/usr/bin/env python3
"""
AI Mango Orchard Health & Pest Intelligence — Full Pipeline Validation Runner
Executes multi-modal assessment on real sample imagery and outputs complete report.
"""

import sys
import json
from pathlib import Path

# Ensure repository root is on sys.path
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ml.pipeline import MangoOrchardPipeline

def run_pipeline_demo():
    print("=" * 80)
    print("🥭 AI Mango Orchard Health & Pest Intelligence — End-to-End Diagnostic Suite")
    print("=" * 80)

    pipeline = MangoOrchardPipeline()

    leaf_sample = REPO_DIR / "ml" / "datasets" / "sample_leaves" / "anthracnose_sample.jpg"
    pest_sample = REPO_DIR / "ml" / "datasets" / "sample_pests" / "fruit_fly_sample.jpg"
    unreliable_pest_sample = REPO_DIR / "ml" / "datasets" / "sample_pests" / "mango_hopper_sample.jpg"

    if not leaf_sample.exists():
        print(f"Leaf sample {leaf_sample} not found. Running dataset generator...")
        from ml.datasets.generate_sample_data import main as gen_data
        gen_data()

    print(f"\n[1/2] 🌿 Running Full Orchard Diagnostic (Confirmed Pest Scenario)...")
    print(f"  Leaf Input: {leaf_sample.name}")
    print(f"  Pest Input: {pest_sample.name}")
    print(f"  Coordinates: Krishnagiri Mango Cluster (12.5266°N, 78.2147°E)")

    result_confirmed = pipeline.analyze_orchard(
        leaf_image=leaf_sample,
        pest_image=pest_sample,
        latitude=12.5266,
        longitude=78.2147
    )

    d_info = result_confirmed["disease_diagnosis"]
    p_info = result_confirmed["pest_intelligence"]
    ohi_info = result_confirmed["orchard_health_index"]
    weather = result_confirmed["weather_vulnerability"]
    recs = result_confirmed["treatment_protocols"]

    print("\n" + "-" * 50)
    print("📋 DIAGNOSTIC RESULTS SUMMARY")
    print("-" * 50)
    print(f"🍃 Disease Diagnosis:     {d_info['disease_name']} (Confidence: {d_info['confidence']*100:.1f}%)")
    print(f"🔍 Lesion Surface Area:   {d_info['lesion_severity_percentage']:.2f}% of leaf blade")
    if d_info.get('gradcam_overlay_path'):
        print(f"🎯 Grad-CAM Explainable:  {Path(d_info['gradcam_overlay_path']).name}")

    print(f"\n🦗 Pest Intelligence:     {p_info['total_detections']} detected (Confirmed: {p_info['confirmed_count']}, Unreliable: {p_info['unreliable_count']})")
    for det in p_info['detections']:
        status_tag = "⚠️ UNVERIFIED" if det['is_unreliable'] else "✅ CONFIRMED"
        print(f"   - {det['pest_name']} ({det['confidence']*100:.1f}%) [{status_tag}]")

    print(f"\n🌤️ Weather (Open-Meteo):   {weather['current_temperature_c']}°C, {weather['current_relative_humidity_pct']}% RH, {weather['current_precipitation_mm']}mm rain")
    if weather['7_day_forecast']:
        next_day = weather['7_day_forecast'][0]
        print(f"   - Tomorrow Outbreak:   Anthracnose Risk {next_day.get('anthracnose_spread_risk')}%, Alert: {next_day.get('alert_level')}")

    print(f"\n⭐ ORCHARD HEALTH INDEX:  {ohi_info['ohi_score']} / 100 — {ohi_info['health_status']}")
    print(f"   - Penalties:           {ohi_info['penalties']}")
    print(f"   - Nutrient Defaulted:  {ohi_info['nutrient_input_was_defaulted']}")
    print(f"   - Disclosed Caveats:   {ohi_info['caveats']}")

    print(f"\n💊 Source-Cited Protocols: {recs['total_protocols']} verified management practices found")
    for r in recs['protocols'][:3]:
        print(f"   - [{r['treatment_type']}] {r['title']} -> {r.get('product_name', 'Cultural Method')}")
        print(f"     Dosage: {r['dosage_rate']}")
        print(f"     Citation: {r['citation_source']}")

    # --- Test Case 2: Unreliable Class Gating (Rule 3) ---
    print(f"\n[2/2] ⚠️ Testing Inviolable Rule 3 (Unreliable Class Gating: Mango Hopper)...")
    result_unreliable = pipeline.analyze_orchard(
        leaf_image=leaf_sample,
        pest_image=unreliable_pest_sample,
        latitude=12.5266,
        longitude=78.2147
    )
    p_unrel = result_unreliable["pest_intelligence"]
    print(f"  Unreliable Pest Detected: {p_unrel['has_unreliable_classes']}")
    print(f"  Warning Message: {p_unrel['unreliable_warning']}")

    print("\n" + "=" * 80)
    print("✅ Full Orchard Intelligence Pipeline Validation Succeeded!")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline_demo()
