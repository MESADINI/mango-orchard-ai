#!/usr/bin/env python3
"""
OHI sensitivity analysis (Phase 5). CPU only; no model is run.

    python ml/severity/ohi_sensitivity.py [--n-mc 2000] [--seed 42]

Scenarios: each image of the MangoLeafBD test split (n = 239) is one scenario.
  disease  deployed classifier's saved fp32 test logits (results/disease/final/efficientnet_b3/logits_test.npz),
           T = 1 and the deployed low-confidence threshold: below it the disease component is excluded, as in the app
  severity HSV segmenter (ml/severity/severity.py) on the same image
  pest     detections = the ground-truth boxes of one randomly drawn pest test image (confidence 1); the pest images
           are not mango (traps, grape, coffee), so pairing is synthetic and only exercises the formula's range
  weather  one randomly drawn hour of NASA POWER data for Krishnagiri (cached by ml/forecasting/backtest.py)
  nutrient never measured -> excluded, as in the app
Analyses (OHI computed by the production engine, ml/severity/ohi.py):
  1. one-at-a-time: each component weight x {0.8, 0.9, 1.1, 1.2} (then renormalised), and each virulence / pest
     impact factor x {0.8, 1.2}
  2. Monte Carlo: Dirichlet weights centred on the nominal weights (concentration 100 = moderate, 20 = wide)
     and uniform Dirichlet(1, 1, 1) (no prior on the weights)
  3. component ablation: each available component removed in turn
Per perturbation: mean / 95th percentile / max |delta OHI|, fraction of scenarios whose health tier changes, and
Spearman rho between perturbed and nominal scores.
Outputs: results/ohi/{sensitivity.json, sensitivity.md, scenarios.csv, oat.png}
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.severity.ohi import (  # noqa: E402
    DEFAULT_WEIGHTS, PEST_IMPACT_FACTORS, VIRULENCE_FACTORS, OrchardHealthIndexEngine, tier,
)
from ml.utils.experiment import collect_env, write_json  # noqa: E402

RESULTS = REPO_ROOT / "results" / "ohi"
DISEASE_RUN = REPO_ROOT / "results" / "disease" / "final" / "efficientnet_b3"
DEPLOYED = REPO_ROOT / "results" / "disease" / "deployed.json"
PEST_SPLIT = REPO_ROOT / "ml" / "datasets" / "processed" / "pest" / "split.csv"
PEST_LABELS = REPO_ROOT / "ml" / "datasets" / "processed" / "pest" / "labels" / "test"
PEST_NAMES = ["Fruit Fly", "Mealy Bug", "Scale Insect"]
WEATHER_CACHE = REPO_ROOT / "ml" / "datasets" / "raw" / "weather"
KRISHNAGIRI = (12.5266, 78.2147)
OAT_FACTORS = (0.8, 0.9, 1.1, 1.2)
TABLE_FACTORS = (0.8, 1.2)
DIRICHLET = {"moderate (c=100)": 100.0, "wide (c=20)": 20.0}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def weather_hours(cache_dir: Path = WEATHER_CACHE) -> Dict:
    """All cached NASA POWER hours (complete records only) and the SHA-256 of every cache file used."""
    from ml.forecasting.weather import fetch_nasa_power_hourly
    files = sorted(Path(cache_dir).glob(f"nasa_power_{KRISHNAGIRI[0]}_{KRISHNAGIRI[1]}_*.json"))
    if not files:
        raise SystemExit("No cached NASA POWER data; run ml/forecasting/backtest.py first.")
    stamps = sorted((f.stem.split("_")[-2], f.stem.split("_")[-1]) for f in files)
    start, end = stamps[0][0], stamps[-1][1]
    data = fetch_nasa_power_hourly(*KRISHNAGIRI, f"{start[:4]}-{start[4:6]}-{start[6:]}",
                                   f"{end[:4]}-{end[4:6]}-{end[6:]}", cache_dir=cache_dir)
    if data is None:
        raise SystemExit("Cached weather is incomplete (a quarter is missing).")
    h = data["hourly"]
    rows = [(t, rh, tc, p) for t, tc, rh, p in zip(h["time"], h["temperature_2m"], h["relative_humidity_2m"],
                                                   h["precipitation"]) if None not in (tc, rh, p)]
    return {"rows": rows, "period": [h["time"][0], h["time"][-1]], "sha256": data["_sha256"],
            "source": data["_source"]}


def pest_scenarios() -> List[List[dict]]:
    """Ground-truth boxes of every pest test image as detection lists (confidence 1; empty = background image)."""
    with open(PEST_SPLIT, encoding="utf-8") as f:
        images = [r["image"] for r in csv.DictReader(f) if r["split"] == "test"]
    out = []
    for img in images:
        lbl = PEST_LABELS / (Path(img).stem + ".txt")
        lines = lbl.read_text().split() if lbl.exists() else []
        classes = [int(v) for v in lines[0::5]]
        out.append([{"pest_name": PEST_NAMES[c], "confidence": 1.0, "is_unreliable": False} for c in classes])
    return out


def build_scenarios(seed: int = 42) -> Dict:
    from PIL import Image
    from ml.severity.severity import compute_lesion_severity
    from ml.disease_detection.inference import DISEASE_CLASSES

    d = np.load(DISEASE_RUN / "logits_test.npz")
    deployed = json.loads(DEPLOYED.read_text(encoding="utf-8"))
    T, thr = float(deployed.get("temperature", 1.0)), float(deployed["ood_threshold"])
    probs = _softmax(d["logits"] / T)
    weather, pests = weather_hours(), pest_scenarios()
    rng = random.Random(seed)
    scenarios = []
    for i, path in enumerate(d["path"]):
        img = np.array(Image.open(REPO_ROOT / str(path)).convert("RGB"))
        sev = compute_lesion_severity(img)["severity_percentage"]
        k = int(probs[i].argmax())
        t, rh, tc, p = rng.choice(weather["rows"])
        pest_idx = rng.randrange(len(pests))
        scenarios.append({
            "image": str(path), "true_label": DISEASE_CLASSES[int(d["y"][i])], "disease_name": DISEASE_CLASSES[k],
            "disease_confidence": float(probs[i, k]), "low_confidence": bool(probs[i, k] < thr),
            "severity_pct": float(sev), "pest_index": pest_idx, "pests": pests[pest_idx],
            "weather_time": t, "relative_humidity": rh, "temperature": tc, "precipitation_mm": p})
    meta = {"n": len(scenarios), "temperature": T, "low_confidence_threshold": thr,
            "weather_period": weather["period"], "weather_source": weather["source"],
            "weather_cache_sha256": weather["sha256"], "n_weather_hours": len(weather["rows"]),
            "n_pest_images": len(pests), "logits_file": str((DISEASE_RUN / "logits_test.npz").relative_to(REPO_ROOT))}
    return {"scenarios": scenarios, "meta": meta}


def score_all(scenarios: List[dict], engine: OrchardHealthIndexEngine, drop: Optional[str] = None) -> np.ndarray:
    out = []
    for s in scenarios:
        missing = [] if not s["low_confidence"] else ["disease"]
        if drop:
            missing.append(drop)
        r = engine.compute_ohi(
            disease_name=None if "disease" in missing else s["disease_name"],
            disease_confidence=s["disease_confidence"], disease_severity_pct=s["severity_pct"],
            pest_detections=s["pests"], relative_humidity=s["relative_humidity"], temperature=s["temperature"],
            precipitation_mm=s["precipitation_mm"], missing=sorted(set(missing)))
        out.append(np.nan if r["ohi_score"] is None else r["ohi_score"])
    return np.array(out, dtype=float)


def compare(nominal: np.ndarray, perturbed: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    ok = ~np.isnan(nominal) & ~np.isnan(perturbed)
    a, b = nominal[ok], perturbed[ok]
    delta = np.abs(b - a)
    tiers_changed = np.mean([tier(x)[0] != tier(y)[0] for x, y in zip(a, b)])
    rho = float(spearmanr(a, b).statistic) if len(a) > 2 and np.ptp(a) > 0 and np.ptp(b) > 0 else float("nan")
    return {"n": int(ok.sum()), "mean_abs_delta": float(delta.mean()), "p95_abs_delta": float(np.percentile(delta, 95)),
            "max_abs_delta": float(delta.max()), "tier_change_fraction": float(tiers_changed), "spearman_rho": rho}


def engine_with(weights=None, virulence=None, pest_impact=None) -> OrchardHealthIndexEngine:
    w = weights or DEFAULT_WEIGHTS
    return OrchardHealthIndexEngine(w["disease"], w["pest"], w["weather"], w["nutrient"],
                                    virulence=virulence, pest_impact=pest_impact)


def analyse(scenarios: List[dict], n_mc: int = 2000, seed: int = 42) -> dict:
    nominal = score_all(scenarios, engine_with())
    res = {"nominal": {"mean": float(np.nanmean(nominal)), "sd": float(np.nanstd(nominal)),
                       "min": float(np.nanmin(nominal)), "max": float(np.nanmax(nominal)),
                       "tier_counts": {}}}
    for x in nominal[~np.isnan(nominal)]:
        res["nominal"]["tier_counts"][tier(x)[0]] = res["nominal"]["tier_counts"].get(tier(x)[0], 0) + 1

    oat = {}
    for comp in ("disease", "pest", "weather"):                # nutrient is never measured, so never weighted
        for f in OAT_FACTORS:
            w = dict(DEFAULT_WEIGHTS)
            w[comp] *= f
            oat[f"weight:{comp}x{f}"] = compare(nominal, score_all(scenarios, engine_with(weights=w)))
    for name in [n for n in VIRULENCE_FACTORS if n != "Healthy"]:
        for f in TABLE_FACTORS:
            v = dict(VIRULENCE_FACTORS)
            v[name] *= f
            oat[f"virulence:{name}x{f}"] = compare(nominal, score_all(scenarios, engine_with(virulence=v)))
    for name in PEST_NAMES:
        for f in TABLE_FACTORS:
            p = dict(PEST_IMPACT_FACTORS)
            p[name] *= f
            oat[f"pest_impact:{name}x{f}"] = compare(nominal, score_all(scenarios, engine_with(pest_impact=p)))
    res["one_at_a_time"] = oat

    rng = np.random.default_rng(seed)
    comps = ("disease", "pest", "weather")
    base = np.array([DEFAULT_WEIGHTS[c] for c in comps])
    base = base / base.sum()
    mc = {}
    for label, alpha in {**{k: base * c for k, c in DIRICHLET.items()}, "uniform Dirichlet(1,1,1)": np.ones(3)}.items():
        draws = rng.dirichlet(alpha, size=n_mc)
        stats = []
        for w3 in draws:
            w = {**dict(zip(comps, w3.tolist())), "nutrient": 0.0}
            stats.append(compare(nominal, score_all(scenarios, engine_with(weights=w))))
        mc[label] = {"n_draws": n_mc, **{k: {"median": float(np.median([s[k] for s in stats])),
                                             "p05": float(np.percentile([s[k] for s in stats], 5)),
                                             "p95": float(np.percentile([s[k] for s in stats], 95))}
                                         for k in ("mean_abs_delta", "tier_change_fraction", "spearman_rho")},
                     "weight_quantiles": {c: [float(np.percentile(draws[:, i], q)) for q in (5, 50, 95)]
                                          for i, c in enumerate(comps)}}
    res["monte_carlo"] = mc
    res["ablation"] = {f"without {c}": compare(nominal, score_all(scenarios, engine_with(), drop=c)) for c in comps}
    return res, nominal


def write_outputs(built: Dict, res: Dict, nominal: np.ndarray, out_dir: Path = RESULTS, seed: int = 42,
                  n_mc: int = 2000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "sensitivity.json", {"meta": built["meta"], "seed": seed, "n_mc": n_mc, "results": res,
                                              "env": collect_env()})
    with open(out_dir / "scenarios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_label", "predicted", "confidence", "low_confidence", "severity_pct", "n_pests",
                    "weather_time", "rh", "temp_c", "precip_mm", "ohi_nominal"])
        for s, o in zip(built["scenarios"], nominal):
            w.writerow([s["image"], s["true_label"], s["disease_name"], f"{s['disease_confidence']:.4f}",
                        s["low_confidence"], f"{s['severity_pct']:.2f}", len(s["pests"]), s["weather_time"],
                        s["relative_humidity"], s["temperature"], s["precipitation_mm"],
                        "" if np.isnan(o) else f"{o:.2f}"])

    def row(name, c):
        return (f"| {name} | {c['mean_abs_delta']:.2f} | {c['p95_abs_delta']:.2f} | {c['max_abs_delta']:.2f} | "
                f"{c['tier_change_fraction']:.3f} | {c['spearman_rho']:.4f} |")
    head = ["| Perturbation | mean abs. dOHI | p95 | max | Tier changed | Spearman rho |", "|---|---|---|---|---|---|"]
    n = res["nominal"]
    md = [f"# OHI sensitivity analysis\n",
          f"Scenarios: n = {built['meta']['n']} (MangoLeafBD test split; deployed classifier predictions; synthetic "
          f"pest and weather pairing, see ml/severity/ohi_sensitivity.py). Nominal OHI mean {n['mean']:.1f} "
          f"(SD {n['sd']:.1f}, range {n['min']:.1f}–{n['max']:.1f}); tiers {n['tier_counts']}.\n",
          "## One-at-a-time\n", *head, *[row(k, v) for k, v in res["one_at_a_time"].items()],
          "\n## Monte Carlo weights (median [5th, 95th percentile] over draws)\n",
          "| Distribution | mean abs. dOHI | Tier changed | Spearman rho |", "|---|---|---|---|"]
    for k, v in res["monte_carlo"].items():
        md.append(f"| {k} | " + " | ".join(
            f"{v[m]['median']:.3f} [{v[m]['p05']:.3f}, {v[m]['p95']:.3f}]"
            for m in ("mean_abs_delta", "tier_change_fraction", "spearman_rho")) + " |")
    md += ["\n## Component ablation\n", *head, *[row(k, v) for k, v in res["ablation"].items()], ""]
    (out_dir / "sensitivity.md").write_text("\n".join(md), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in res["one_at_a_time"] if k.startswith("weight:")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(keys, [res["one_at_a_time"][k]["mean_abs_delta"] for k in keys], color="#4C8C4A")
    ax.set_xlabel("mean |ΔOHI| (points)")
    ax.set_title("OHI one-at-a-time weight sensitivity")
    fig.tight_layout()
    fig.savefig(out_dir / "oat.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-mc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    built = build_scenarios(a.seed)
    res, nominal = analyse(built["scenarios"], a.n_mc, a.seed)
    write_outputs(built, res, nominal, seed=a.seed, n_mc=a.n_mc)
    print((RESULTS / "sensitivity.md").read_text(encoding="utf-8"))
