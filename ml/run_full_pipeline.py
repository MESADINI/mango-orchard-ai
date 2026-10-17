#!/usr/bin/env python3
"""
AI Mango Orchard Health & Pest Intelligence — Full Pipeline Runner
Runs the multi-modal assessment on sample imagery (or your own images) and prints a summary.

Usage:
    python ml/run_full_pipeline.py [--leaf PATH] [--pest PATH] [--lat 12.5266] [--lon 78.2147]
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ml.pipeline import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, MangoOrchardPipeline

# Emoji output must not crash consoles that are not UTF-8 (e.g. redirected output on Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

SAMPLE_LEAF = REPO_DIR / "ml" / "datasets" / "sample_leaves" / "anthracnose_sample.jpg"
SAMPLE_PEST = REPO_DIR / "ml" / "datasets" / "sample_pests" / "fruit_fly_sample.jpg"


def run_pipeline_demo(leaf: Path, pest: Path, lat: float, lon: float):
    print("=" * 80)
    print("🥭 AI Mango Orchard Health & Pest Intelligence — End-to-End Diagnostic")
    print("=" * 80)
    print(f"  Leaf Input:  {leaf}")
    print(f"  Pest Input:  {pest}")
    print(f"  Coordinates: {lat}, {lon}")

    result = MangoOrchardPipeline().analyze_orchard(leaf_image=leaf, pest_image=pest, latitude=lat, longitude=lon)
    d_info = result["disease_diagnosis"]
    p_info = result["pest_intelligence"]
    ohi_info = result["orchard_health_index"]
    weather = result["weather_vulnerability"]
    recs = result["treatment_protocols"]

    print("\n" + "-" * 50)
    print("📋 DIAGNOSTIC RESULTS SUMMARY")
    print("-" * 50)
    if d_info["model_available"]:
        print(f"🍃 Disease Diagnosis:     {d_info['disease_name']} (Confidence: {d_info['confidence'] * 100:.1f}%)")
    else:
        print("🍃 Disease Diagnosis:     UNAVAILABLE (no trained EfficientNet-B3 weights)")
    print(f"🔍 Lesion Surface Area:   {d_info['lesion_severity_percentage']:.2f}% of leaf blade (colour heuristic)")
    if d_info.get("gradcam_overlay_path"):
        print(f"🎯 Grad-CAM overlay:      {d_info['gradcam_overlay_path']}")

    if p_info["model_available"]:
        print(f"\n🦗 Pest Detections:       {p_info['total_detections']} "
              f"(Confirmed: {p_info['confirmed_count']}, Unverified: {p_info['unreliable_count']})")
        for det in p_info["detections"]:
            tag = "⚠️ UNVERIFIED" if det["is_unreliable"] else "✅ CONFIRMED"
            print(f"   - {det['pest_name']} ({det['confidence'] * 100:.1f}%) [{tag}]")
    else:
        print("\n🦗 Pest Detection:        UNAVAILABLE (no trained YOLOv8 weights)")

    source = "live Open-Meteo" if weather["live_data"] else "defaults - offline"
    print(f"\n🌤️ Weather ({source}): {weather['current_temperature_c']}°C, "
          f"{weather['current_relative_humidity_pct']}% RH, {weather['current_precipitation_mm']} mm rain")
    if weather["7_day_forecast"]:
        day = weather["7_day_forecast"][0]
        print(f"   - Today: anthracnose {day['anthracnose_risk']}, powdery mildew {day['powdery_mildew_risk']}, "
              f"bacterial canker {day['bacterial_canker_risk']} (weather vs published favourable conditions)")

    partial = f" (PARTIAL: missing {', '.join(ohi_info['missing_components'])})" if ohi_info["is_partial"] else ""
    print(f"\n⭐ ORCHARD HEALTH INDEX:  {ohi_info['ohi_score']} / 100 — {ohi_info['health_status']}{partial}")
    print(f"   - Penalties: {ohi_info['penalties']}")
    for caveat in ohi_info["caveats"]:
        print(f"   - Caveat: {caveat}")

    print(f"\n💊 Source-Cited Protocols: {recs['total_protocols']}")
    for r in recs["protocols"]:
        print(f"   - [{r['treatment_type']}] {r['condition']}: {r['title']} -> {r['product_name']}")
        print(f"     Dosage: {r['dosage_rate']} | Citation: {r['citation_source']}")

    print("\n" + "=" * 80)
    print("✅ Pipeline run complete.")
    print("=" * 80)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full orchard diagnostic pipeline")
    parser.add_argument("--leaf", type=Path, default=SAMPLE_LEAF)
    parser.add_argument("--pest", type=Path, default=SAMPLE_PEST)
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE)
    args = parser.parse_args()
    run_pipeline_demo(args.leaf, args.pest, args.lat, args.lon)
