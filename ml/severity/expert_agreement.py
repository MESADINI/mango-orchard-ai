#!/usr/bin/env python3
"""
OHI vs. expert judgement (Phase 5 scaffolding; no expert ratings exist yet).

    python ml/severity/expert_agreement.py template [--n 40]     -> ml/datasets/annotation/ohi_expert_ratings_template.csv
    python ml/severity/expert_agreement.py evaluate FILE [FILE...] -> results/ohi/expert_agreement.{json,md}

template: draws n scenarios from results/ohi/scenarios.csv (written by ohi_sensitivity.py), stratified by nominal OHI
          tier, and writes a blind rating sheet: the inputs an agronomist would see (leaf image, predicted disease,
          lesion %, pest count, weather), never the OHI. Each rater fills `rating_0_100` (100 = fully healthy) and
          returns their own copy with `rater_id` set. Instructions: docs/annotation/OHI_EXPERT_RATING.md.
evaluate: Spearman rho between the OHI and the mean expert rating, with a scenario-level bootstrap 95% CI; with two
          or more raters also mean pairwise inter-rater Spearman (the ceiling the OHI can be expected to reach).
          Requires at least MIN_RATED scenarios with a rating.
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.severity.ohi import tier  # noqa: E402
from ml.utils.experiment import write_json  # noqa: E402

SCENARIOS = REPO_ROOT / "results" / "ohi" / "scenarios.csv"
TEMPLATE = REPO_ROOT / "ml" / "datasets" / "annotation" / "ohi_expert_ratings_template.csv"
RESULTS = REPO_ROOT / "results" / "ohi"
MIN_RATED = 20
SHOWN = ["image", "predicted", "confidence", "severity_pct", "n_pests", "rh", "temp_c", "precip_mm"]
FIELDS = ["scenario_id", *SHOWN, "rater_id", "rating_0_100", "notes"]


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_template(n: int = 40, seed: int = 42, scenarios: Path = SCENARIOS, out: Path = TEMPLATE) -> Path:
    # scenario_id = row index in scenarios.csv (as in evaluate); scenarios without an OHI are not offered
    rows = [{**r, "scenario_id": str(i)} for i, r in enumerate(_read(scenarios)) if r["ohi_nominal"]]
    by_tier: Dict[str, List[dict]] = {}
    for r in rows:
        by_tier.setdefault(tier(float(r["ohi_nominal"]))[0], []).append(r)
    rng = random.Random(seed)
    picked, tiers = [], sorted(by_tier)
    while len(picked) < min(n, len(rows)):                    # round-robin over tiers -> stratified
        for t in tiers:
            if by_tier[t] and len(picked) < n:
                picked.append(by_tier[t].pop(rng.randrange(len(by_tier[t]))))
    rng.shuffle(picked)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in picked:
            w.writerow({**{k: r[k] for k in ["scenario_id", *SHOWN]}, "rater_id": "", "rating_0_100": "", "notes": ""})
    return out


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def evaluate(files: List[Path], scenarios: Path = SCENARIOS, out_dir: Path = RESULTS, n_boot: int = 2000,
             seed: int = 42) -> dict:
    ohi = {str(i): float(r["ohi_nominal"]) for i, r in enumerate(_read(scenarios)) if r["ohi_nominal"]}
    ratings: Dict[str, Dict[str, float]] = {}                  # rater -> scenario -> rating
    for path in files:
        for r in _read(path):
            if r.get("rating_0_100", "").strip():
                v = float(r["rating_0_100"])
                if not 0 <= v <= 100:
                    raise ValueError(f"{path}: rating {v} outside 0-100 (scenario {r['scenario_id']})")
                ratings.setdefault(r["rater_id"].strip() or Path(path).stem, {})[r["scenario_id"]] = v
    rated = sorted({s for per in ratings.values() for s in per if s in ohi}, key=int)
    if len(rated) < MIN_RATED:
        raise SystemExit(f"Only {len(rated)} rated scenarios; at least {MIN_RATED} are needed.")
    mean_rating = np.array([np.mean([per[s] for per in ratings.values() if s in per]) for s in rated])
    score = np.array([ohi[s] for s in rated])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rated), len(rated))
        if np.ptp(score[idx]) > 0 and np.ptp(mean_rating[idx]) > 0:
            boot.append(_spearman(score[idx], mean_rating[idx]))
    res = {"n_scenarios": len(rated), "n_raters": len(ratings), "spearman_ohi_vs_mean_rating": _spearman(score, mean_rating),
           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n_boot": n_boot, "seed": seed}
    raters = sorted(ratings)
    pairs = [_spearman([ratings[a][s] for s in rated if s in ratings[a] and s in ratings[b]],
                       [ratings[b][s] for s in rated if s in ratings[a] and s in ratings[b]])
             for i, a in enumerate(raters) for b in raters[i + 1:]
             if sum(s in ratings[a] and s in ratings[b] for s in rated) >= 3]
    res["inter_rater_mean_spearman"] = float(np.mean(pairs)) if pairs else None
    write_json(Path(out_dir) / "expert_agreement.json", res)
    (Path(out_dir) / "expert_agreement.md").write_text(
        f"OHI vs. mean expert rating: Spearman rho = {res['spearman_ohi_vs_mean_rating']:.3f} "
        f"[95% CI {res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}], {res['n_scenarios']} scenarios, "
        f"{res['n_raters']} rater(s). Inter-rater: {res['inter_rater_mean_spearman']}\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--n", type=int, default=40)
    e = sub.add_parser("evaluate")
    e.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    if a.cmd == "template":
        print(make_template(a.n))
    else:
        print(evaluate(a.files))
