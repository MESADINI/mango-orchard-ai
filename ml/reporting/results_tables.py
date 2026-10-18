#!/usr/bin/env python3
"""
Results tables for the README and the paper, generated only from files under results/ (never typed by hand).

    python tasks.py paper-tables      (= python ml/reporting/results_tables.py)

Writes
  docs/RESULTS.md     every generated table, each with its source file
  README.md           the headline table between <!-- RESULTS:START --> and <!-- RESULTS:END -->
A result that has not been produced yet is shown as "pending (<source file>)", never as a number.
"""

import json
import re
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
README = REPO_ROOT / "README.md"
RESULTS_MD = REPO_ROOT / "docs" / "RESULTS.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

# (section title, generated markdown table under results/)
TABLES = [
    ("Disease classifier: 5-fold group cross-validation", "disease/tables/cv_comparison.md"),
    ("Disease classifier: paired significance tests", "disease/tables/significance.md"),
    ("Disease classifier: final models (test, external, calibration, OOD)", "disease/tables/final_models.md"),
    ("Disease classifier: external set per class (selected model)", "disease/tables/external_selected.md"),
    ("Disease classifier: ablations (fold 0)", "disease/tables/ablation.md"),
    ("Pest detector", "pest/models.md"),
    ("Lesion severity vs expert masks", "severity/summary.md"),
    ("Grad-CAM faithfulness", "xai/faithfulness.md"),
    ("OHI sensitivity analysis", "ohi/sensitivity.md"),
    ("OHI vs expert ratings", "ohi/expert_agreement.md"),
    ("Forecaster back-test (weather only)", "forecaster/monthly.md"),
    ("API latency", "api/latency.md"),
]


def _json(rel: str) -> Optional[dict]:
    p = RESULTS / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pending(rel: str) -> str:
    return f"pending (`results/{rel}`)"


def headline_rows() -> List[Tuple[str, str, str, str]]:
    """(component, metric, value, source) rows; each value is read from results/ or marked pending."""
    rows = []

    def add(component: str, metric: str, rel: str, fn: Callable[[dict], str]):
        data = _json(rel)
        try:
            value = fn(data) if data is not None else _pending(rel)
        except (KeyError, TypeError, IndexError):
            value = _pending(rel)
        rows.append((component, metric, value, f"`results/{rel}`"))

    def sel(d):
        return d["selected"]

    add("Disease classifier", "5-fold group CV macro-F1 (selected model, mean ± SD)", "disease/summary.json",
        lambda d: f"{d['cv'][sel(d)]['macro_f1']['mean']:.3f} ± {d['cv'][sel(d)]['macro_f1']['sd']:.3f} ({sel(d)})")
    add("Disease classifier", "Test-split accuracy (n = 239)", "disease/summary.json",
        lambda d: f"{d['final'][sel(d)]['test']['accuracy']:.3f}")
    add("Disease classifier", "**External set macro-F1 [95% CI]** (independent images)", "disease/summary.json",
        lambda d: "**{:.3f} [{:.3f}, {:.3f}]**".format(d["final"][sel(d)]["external"]["macro_f1"],
                                                      *d["final"][sel(d)]["external"]["ci95_macro_f1"]))
    add("Disease classifier", "External images flagged low-confidence (retake photo)", "disease/summary.json",
        lambda d: f"{d['final'][sel(d)]['external']['rejected_by_ood_threshold']:.1%}")
    add("Pest detector", "Test mAP50-95 (selected by val; non-mango proxy data)", "pest/deployed.json",
        lambda d: f"{d['test_map50_95']:.3f} ({d['model']})")
    add("Lesion severity", "Lesion IoU vs expert masks", "severity/validation.json",
        lambda d: f"{d['overall']['lesion_iou']['mean']:.3f} (n = {d['n_masks']})")
    add("Grad-CAM", "Deletion AUC: Grad-CAM vs centre prior (lower is better)", "xai/faithfulness.json",
        lambda d: "{:.3f} vs {:.3f}".format(d["summary"]["deletion"]["gradcam"]["mean_auc"],
                                            d["summary"]["deletion"]["centre"]["mean_auc"]))
    add("OHI", "Max tier change, one weight ±20%", "ohi/sensitivity.json",
        lambda d: f"{max(v['tier_change_fraction'] for k, v in d['results']['one_at_a_time'].items() if k.startswith('weight:')):.1%}")
    add("OHI", "Agreement with expert ratings (Spearman)", "ohi/expert_agreement.json",
        lambda d: f"{d['spearman_ohi_vs_mean_rating']:.3f} (n = {d['n_scenarios']})")
    add("Forecaster", "Days/year meeting documented 'High' conditions (Krishnagiri, 2015–2024)",
        "forecaster/backtest.json",
        lambda d: ", ".join(f"{k.replace('_', ' ')} {v:.1f}" for k, v in d["documented_conditions_days_per_year"].items()))
    add("API", "Median POST /api/analyze latency, dual scan, CPU", "api/latency.json",
        lambda d: f"{d['scan_types']['dual']['median_ms']:.0f} ms")
    return rows


def headline_table() -> str:
    lines = ["| Component | Metric | Value | Source |", "|---|---|---|---|"]
    lines += [f"| {c} | {m} | {v} | {s} |" for c, m, v, s in headline_rows()]
    lines += ["", "Generated by `python tasks.py paper-tables` from `results/`; full tables in "
              "[docs/RESULTS.md](docs/RESULTS.md). Outbreak prediction, field accuracy and expert agreement have "
              "not been validated (see Known limitations)."]
    return "\n".join(lines)


def results_md() -> str:
    out = ["# Results", "",
           "Generated by `python tasks.py paper-tables`. Each table is copied verbatim from the file named under it; "
           "nothing here is typed by hand.", "", "## Headline", "", headline_table(), ""]
    for title, rel in TABLES:
        p = RESULTS / rel
        out += [f"## {title}", ""]
        out += [p.read_text(encoding="utf-8").strip() if p.exists() else f"_{_pending(rel)}_", "",
                f"Source: `results/{rel}`", ""]
    return "\n".join(out)


LATEX_NUMBERS = REPO_ROOT / "paper" / "generated" / "numbers.tex"


def latex_numbers() -> str:
    r"""\newcommand macros for the paper; a value that is not in results/ yet expands to \pending{<source>}."""
    defs = []
    pct = r"\%"

    def macro(name: str, rel: str, fn: Callable[[dict], str]):
        data = _json(rel)
        try:
            value = fn(data) if data is not None else None
        except (KeyError, TypeError, IndexError):
            value = None
        body = value if value is not None else r"\pending{" + rel.replace("_", r"\_") + "}"
        defs.append(r"\newcommand{\%s}{%s}" % (name, body))

    def sel(d):
        return d["selected"]

    def fin(d):
        return d["final"][sel(d)]

    macro("cvMacroFOne", "disease/summary.json",
          lambda d: "%.3f $\\pm$ %.3f" % (d["cv"][sel(d)]["macro_f1"]["mean"], d["cv"][sel(d)]["macro_f1"]["sd"]))
    macro("testAcc", "disease/summary.json", lambda d: "%.3f" % fin(d)["test"]["accuracy"])
    macro("extMacroFOne", "disease/summary.json", lambda d: "%.3f" % fin(d)["external"]["macro_f1"])
    macro("extMacroFOneCI", "disease/summary.json",
          lambda d: "[%.3f, %.3f]" % tuple(fin(d)["external"]["ci95_macro_f1"]))
    macro("extFlagged", "disease/summary.json",
          lambda d: "%.1f%s" % (100 * fin(d)["external"]["rejected_by_ood_threshold"], pct))
    macro("testFlagged", "disease/summary.json",
          lambda d: "%.1f%s" % (100 * fin(d)["test_rejected_by_ood_threshold"], pct))
    macro("aurocNear", "disease/summary.json", lambda d: "%.3f" % fin(d)["ood"]["ood_near"]["auroc"])
    macro("aurocFar", "disease/summary.json", lambda d: "%.3f" % fin(d)["ood"]["ood_far"]["auroc"])
    macro("pestTestMap", "pest/deployed.json", lambda d: "%.3f" % d["test_map50_95"])
    macro("pestModel", "pest/deployed.json", lambda d: d["model"].replace("_", r"\_"))
    macro("sevLesionIoU", "severity/validation.json", lambda d: "%.3f" % d["overall"]["lesion_iou"]["mean"])
    for mode, short in (("deletion", "del"), ("insertion", "ins")):
        for method, suffix in (("gradcam", "Gradcam"), ("centre", "Centre"), ("random", "Random")):
            macro(f"{short}Auc{suffix}", "xai/faithfulness.json",
                  lambda d, mode=mode, method=method: "%.3f" % d["summary"][mode][method]["mean_auc"])
    macro("ohiMaxTierChange", "ohi/sensitivity.json",
          lambda d: "%.1f%s" % (100 * max(v["tier_change_fraction"] for k, v in d["results"]["one_at_a_time"].items()
                                          if k.startswith("weight:")), pct))
    macro("ohiUniformTierChange", "ohi/sensitivity.json",
          lambda d: "%.1f%s" % (100 * d["results"]["monte_carlo"]["uniform Dirichlet(1,1,1)"]["tier_change_fraction"]
                                ["median"], pct))
    macro("fcPrevHighDays", "forecaster/backtest.json", lambda d: "%.1f" % d["previous_rules_high_days_per_year"]["any"])
    macro("apiDualMs", "api/latency.json", lambda d: "%.0f" % d["scan_types"]["dual"]["median_ms"])
    header = ["% Generated by python tasks.py paper-tables from results/. Do not edit.",
              r"\providecommand{\pending}[1]{\textcolor{red}{[pending: #1]}}"]
    return "\n".join(header + defs) + "\n"


LATEX_TABLES = REPO_ROOT / "paper" / "generated" / "tables.tex"
ARCH_LABEL = {"efficientnet_b3": "EfficientNet-B3", "resnet50": "ResNet-50", "mobilenetv3_large": "MobileNetV3-L",
              "yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolov8s": "YOLOv8s"}


def _tex_table(label: str, caption: str, cols: str, header: str, rows: List[str]) -> str:
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption, r"\label{%s}" % label,
                      r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{%s}" % cols, r"\toprule",
                      header + r" \\", r"\midrule", *[r + r" \\" for r in rows], r"\bottomrule", r"\end{tabular}",
                      r"\end{table}", ""])


def _pending_table(label: str, caption: str, rel: str) -> str:
    return _tex_table(label, caption, "l", "Status", [r"\pending{%s}" % rel.replace("_", r"\_")])


def latex_tables() -> str:
    """LaTeX tables for the paper, built only from results/ (a missing source gives a pending placeholder)."""
    out = ["% Generated by python tasks.py paper-tables from results/. Do not edit."]
    d = _json("disease/summary.json")
    if d:
        rows = []
        for arch in d["study_archs"]:
            cv = d["cv"][arch]
            lat = d["latency"].get(arch, {})
            name = ARCH_LABEL.get(arch, arch) + (r"$^\dagger$" if arch == d["selected"] else "")
            rows.append("%s & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f & %.1f & %.1f" % (
                name, cv["accuracy"]["mean"], cv["accuracy"]["sd"], cv["macro_f1"]["mean"], cv["macro_f1"]["sd"],
                lat.get("params_m", float("nan")), lat.get("median_ms", float("nan"))))
        out.append(_tex_table("tab:cv", r"Five-fold group cross-validation on MangoLeafBD (mean $\pm$ SD over folds). "
                              r"$^\dagger$selected by mean macro-F1. Latency: CPU, batch 1.",
                              "lcccc", r"Model & Accuracy & Macro-F1 & Params (M) & ms", rows))
        rows = []
        for arch in d["study_archs"]:
            f = d["final"][arch]
            rows.append("%s & %.3f & %.3f [%.3f, %.3f] & %.3f / %.3f & %.3f / %.3f & %.2f" % (
                ARCH_LABEL.get(arch, arch), f["test"]["accuracy"], f["external"]["macro_f1"],
                *f["external"]["ci95_macro_f1"], f["calibration_test"]["ece_after"],
                f["calibration_external"]["ece_deployed_T"], f["ood"]["ood_near"]["auroc"],
                f["ood"]["ood_far"]["auroc"], f["external"]["rejected_by_ood_threshold"]))
        out.append(_tex_table("tab:final", r"Final models: test split ($n{=}239$), independent external set "
                              r"($n{=}381$, 4 classes), calibration (ECE) and OOD detection (AUROC, max softmax). "
                              r"Flag: share of external images below the low-confidence threshold.",
                              "lccccc", r"Model & Test acc. & External macro-F1 [95\% CI] & ECE test/ext. & "
                              r"AUROC near/far & Flag", rows))
        rows = []
        for name, a in d["ablation"].items():
            delta = a.get("delta_macro_f1_vs_baseline")
            p = a.get("mcnemar_vs_baseline", {}).get("p_value")
            rows.append("%s & %.3f [%.3f, %.3f] & %s & %s" % (
                name.replace("_", " "), a["macro_f1"], *a["ci95_macro_f1"],
                "--" if delta is None else "%+.3f" % delta, "--" if p is None else "%.2g" % p))
        out.append(_tex_table("tab:ablation", r"Single-run ablations of the EfficientNet-B3 recipe on CV fold 0 "
                              r"($n{=}279$); McNemar exact test against the baseline.",
                              "lccc", r"Variant & Macro-F1 [95\% CI] & $\Delta$ & $p$", rows))
    else:
        out.append(_pending_table("tab:cv", "Cross-validation", "disease/summary.json"))
    pest_rows = []
    for m in ("yolov8n", "yolo11n", "yolov8s"):
        s = _json(f"pest/{m}/summary.json")
        if s:
            pc = s["test"]["per_class"]
            pest_rows.append("%s & %.1f & %.3f & %.3f & %.3f & %.3f / %.3f / %.3f" % (
                ARCH_LABEL[m], s["n_params"] / 1e6, s["val"]["map50_95"], s["test"]["map50"], s["test"]["map50_95"],
                pc["Fruit Fly"]["ap50_95"], pc["Mealy Bug"]["ap50_95"], pc["Scale Insect"]["ap50_95"]))
    out.append(_tex_table("tab:pest", r"Pest detectors on the proxy dataset (test $n{=}379$). Per class: AP50-95 "
                          r"for Fruit Fly / Mealy Bug / Scale Insect. Each class comes from a single non-mango source.",
                          "lccccc", r"Model & Params (M) & Val mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{50:95}$ & Per class",
                          pest_rows) if pest_rows else _pending_table("tab:pest", "Pest detectors", "pest/"))
    x = _json("xai/faithfulness.json")
    if x:
        s = x["summary"]
        rows = ["%s & %.3f & %.3f & %.3f & %.2f (%.1g)" % (
            mode, s[mode]["gradcam"]["mean_auc"], s[mode]["random"]["mean_auc"], s[mode]["centre"]["mean_auc"],
            s[mode]["gradcam_vs_centre"]["fraction_gradcam_better"], s[mode]["gradcam_vs_centre"]["wilcoxon_p"])
            for mode in ("deletion", "insertion")]
        out.append(_tex_table("tab:xai", r"Grad-CAM faithfulness (test split, $n{=}%d$): mean AUC; last column: share "
                              r"of images where Grad-CAM beats the centre prior (Wilcoxon $p$)." % s["n_images"],
                              "lcccc", r"Curve & Grad-CAM & Random & Centre & Better than centre", rows))
    else:
        out.append(_pending_table("tab:xai", "Grad-CAM faithfulness", "xai/faithfulness.json"))
    o = _json("ohi/sensitivity.json")
    if o:
        r = o["results"]
        oat = [v for k, v in r["one_at_a_time"].items() if k.startswith("weight:")]
        rows = ["One weight $\\times$ 0.8--1.2 & %.3f & %.3f" % (min(v["spearman_rho"] for v in oat),
                                                                  max(v["tier_change_fraction"] for v in oat))]
        for k, v in r["monte_carlo"].items():
            rows.append("Dirichlet %s (median) & %.3f & %.3f" % (k.replace("_", " "), v["spearman_rho"]["median"],
                                                                  v["tier_change_fraction"]["median"]))
        for k, v in r["ablation"].items():
            rows.append("%s & %.3f & %.3f" % (k.capitalize(), v["spearman_rho"], v["tier_change_fraction"]))
        out.append(_tex_table("tab:ohi", r"OHI sensitivity ($n{=}%d$ scenarios): Spearman $\rho$ with the nominal "
                              r"scores (worst case for one-at-a-time) and share of scenarios whose tier changes."
                              % o["meta"]["n"], "lcc", r"Perturbation & $\rho$ & Tier changed", rows))
    else:
        out.append(_pending_table("tab:ohi", "OHI sensitivity", "ohi/sensitivity.json"))
    return "\n".join(out)


def update_readme(text: str, table: str) -> str:
    block = f"{START}\n{table}\n{END}"
    if START in text and END in text:
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    raise ValueError(f"README has no {START} ... {END} block")


def main() -> None:
    RESULTS_MD.write_text(results_md(), encoding="utf-8")
    LATEX_NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    LATEX_NUMBERS.write_text(latex_numbers(), encoding="utf-8")
    LATEX_TABLES.write_text(latex_tables(), encoding="utf-8")
    README.write_text(update_readme(README.read_text(encoding="utf-8"), headline_table()), encoding="utf-8")
    print(f"Wrote {RESULTS_MD.relative_to(REPO_ROOT)} and the README results block")


if __name__ == "__main__":
    sys.exit(main())
