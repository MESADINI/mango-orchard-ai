#!/usr/bin/env python3
"""
Phase 2 analysis: turns the saved logits in results/disease/{cv,final,ablation}/ into every reported number.

    python ml/disease_detection/analysis.py

Pre-registered rules (fixed before any result was seen; see docs/phases/PHASE_2.md):
  * Model selection: the architecture with the highest mean CV macro-F1 is selected. The test, external and OOD
    sets are only reported, never used for selection or tuning.
  * Temperature scaling is fitted on the validation split; ECE is reported on the test split before/after.
  * OOD rejection score = max softmax probability after temperature scaling; the threshold keeps 95% of the
    validation images (in-distribution); it is then applied unchanged to test / ood_near / ood_far.
  * Comparisons: Wilcoxon signed-rank on per-fold macro-F1 (n = 5: smallest attainable two-sided p = 0.0625, so it
    cannot reach 0.05 and is reported for completeness) and exact McNemar on the pooled out-of-fold predictions
    (identical images for every architecture), Holm-adjusted over all pairs.
  * Confidence intervals: 95% percentile bootstrap resampling whole leaves (groups), 2,000 resamples.
Outputs: results/disease/summary.json, results/disease/tables/*.md, results/disease/figures/*.png
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.inference import DISEASE_CLASSES  # noqa: E402
from ml.disease_detection.metrics import (  # noqa: E402
    bootstrap_ci, classification_metrics, expected_calibration_error, fit_temperature, holm_adjust, mcnemar_exact,
    nll, ood_metrics, softmax, temperature_at_bound, temperature_identifiable, wilcoxon_paired,
)
from ml.disease_detection.train import DEFERRED_ARCHS, STUDY_ARCHS as ARCHS  # noqa: E402
from ml.utils.experiment import git_info, write_json  # noqa: E402

RESULTS = REPO_ROOT / "results" / "disease"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
OOD_ACCEPT_RATE = 0.95


def load(run_dir: Path, name: str):
    d = np.load(run_dir / f"logits_{name}.npz", allow_pickle=False)
    return d["logits"].astype(np.float64), d["y"].astype(int), d["path"], d["group"]


def macro_f1(y, p):
    return classification_metrics(y, p, DISEASE_CLASSES)["macro_f1"]


def accuracy(y, p):
    return float((np.asarray(p).argmax(1) == y).mean())


def fmt(m, s=None, digits=3):
    return f"{m:.{digits}f}" if s is None else f"{m:.{digits}f} ± {s:.{digits}f}"


# ----------------------------------------------------------------------------- CV
def analyse_cv():
    out, pooled = {}, {}
    for arch in ARCHS:
        folds = [RESULTS / "cv" / arch / f"fold{k}" for k in range(5)]
        if not all((f / "summary.json").exists() for f in folds):
            continue
        per_fold, ys, ps, paths = [], [], [], []
        for f in folds:
            logits, y, path, _ = load(f, "heldout")
            p = softmax(logits)
            per_fold.append(classification_metrics(y, p, DISEASE_CLASSES))
            ys.append(y), ps.append(p), paths.append(path)
        y, p, path = np.concatenate(ys), np.concatenate(ps), np.concatenate(paths)
        order = np.argsort(path)                      # align images across architectures
        pooled[arch] = (y[order], p[order], path[order])
        keys = ("accuracy", "macro_f1", "balanced_accuracy", "macro_auc_ovr")
        out[arch] = {
            **{k: {"mean": float(np.mean([m[k] for m in per_fold])), "sd": float(np.std([m[k] for m in per_fold], ddof=1)),
                   "per_fold": [m[k] for m in per_fold]} for k in keys},
            "pooled_oof": classification_metrics(y, p, DISEASE_CLASSES),
            "epochs_best": [json.loads((f / "summary.json").read_text())["best_epoch"] for f in folds],
        }
    tests = {}
    archs = list(out)
    pairs = list(itertools.combinations(archs, 2))
    if pairs:
        assert all(np.array_equal(pooled[a][2], pooled[archs[0]][2]) for a in archs), "OOF image sets differ"
        mc = {f"{a} vs {b}": mcnemar_exact(pooled[a][1].argmax(1) == pooled[a][0],
                                           pooled[b][1].argmax(1) == pooled[b][0]) for a, b in pairs}
        wx = {f"{a} vs {b}": wilcoxon_paired(out[a]["macro_f1"]["per_fold"], out[b]["macro_f1"]["per_fold"])
              for a, b in pairs}
        for d in (mc, wx):
            for k, adj in zip(d, holm_adjust([v["p_value"] for v in d.values()])):
                d[k]["p_holm"] = adj
        tests = {"mcnemar_pooled_oof": mc, "wilcoxon_fold_macro_f1": wx}
    selected = max(out, key=lambda a: out[a]["macro_f1"]["mean"]) if out else None
    return out, tests, selected


# ----------------------------------------------------------------------------- final models
def analyse_final(arch):
    run = RESULTS / "final" / arch
    if not (run / "summary.json").exists():
        return None
    lv, yv, _, _ = load(run, "val")
    # Pre-registered: fit T on val. Post-hoc rule (added after the first results, disclosed): if T is not identifiable
    # from val (val perfectly separated, or the fit on the search bound), T = 1 is deployed instead.
    T_fit = fit_temperature(lv, yv)
    identifiable = temperature_identifiable(lv, yv, T_fit)
    T = T_fit if identifiable else 1.0
    res = {"temperature": T, "temperature_fitted": T_fit, "temperature_identifiable": identifiable,
           "temperature_at_bound": temperature_at_bound(T_fit),
           "temperature_rule": "fitted on val; T = 1 if not identifiable (val perfectly separated or fit on the "
                               "search bound) - post-hoc rule, disclosed",
           "val_accuracy": float((lv.argmax(1) == yv).mean()),
           "summary": json.loads((run / "summary.json").read_text())}
    conf_val = softmax(lv, T).max(1)
    threshold = float(np.quantile(conf_val, 1 - OOD_ACCEPT_RATE))
    res["ood_threshold"] = threshold
    lt, yt, _, gt = load(run, "test")
    pt_raw, pt = softmax(lt), softmax(lt, T)
    res["test"] = classification_metrics(yt, pt, DISEASE_CLASSES)
    res["test"]["ci95_accuracy"] = bootstrap_ci(yt, pt, accuracy, groups=gt)
    res["test"]["ci95_macro_f1"] = bootstrap_ci(yt, pt, macro_f1, groups=gt)
    res["calibration_test"] = {
        "ece_before": expected_calibration_error(yt, pt_raw), "ece_after": expected_calibration_error(yt, pt),
        "nll_before": nll(lt, yt), "nll_after": nll(lt, yt, T),
        "ece_with_fitted_T": expected_calibration_error(yt, softmax(lt, T_fit))}
    le, ye, _, ge = load(run, "external")
    pe = softmax(le, T)
    res["calibration_external"] = {
        "ece_deployed_T": expected_calibration_error(ye, pe), "nll_deployed_T": nll(le, ye, T),
        "ece_with_fitted_T": expected_calibration_error(ye, softmax(le, T_fit)),
        "mean_confidence": float(pe.max(1).mean()), "accuracy": float((pe.argmax(1) == ye).mean())}
    res["external"] = classification_metrics(ye, pe, DISEASE_CLASSES)
    res["external"]["ci95_macro_f1"] = bootstrap_ci(ye, pe, macro_f1, groups=ge)
    res["external"]["predicted_healthy_fraction"] = float((pe.argmax(1) == DISEASE_CLASSES.index("Healthy")).mean())
    res["external"]["rejected_by_ood_threshold"] = float((pe.max(1) < threshold).mean())
    s_in = pt.max(1)
    res["ood"] = {}
    for name in ("ood_near", "ood_far"):
        lo, _, _, go = load(run, name)
        res["ood"][name] = ood_metrics(s_in, softmax(lo, T).max(1), threshold)
        res["ood"][name]["predicted_class_distribution"] = {
            c: float(np.mean(softmax(lo, T).argmax(1) == i)) for i, c in enumerate(DISEASE_CLASSES)}
    res["test_rejected_by_ood_threshold"] = float((s_in < threshold).mean())
    return res


# ----------------------------------------------------------------------------- ablations
def analyse_ablation(arch="efficientnet_b3", fold=0):
    """Single-run ablations on CV fold `fold`, each compared with the baseline CV run on the same held-out images
    (exact McNemar on per-image correctness). Single runs: no SD, differences are not averaged over seeds."""
    base_run = RESULTS / "cv" / arch / f"fold{fold}"
    abl = RESULTS / "ablation" / arch
    if not (base_run / "summary.json").exists() or not abl.exists():
        return {}
    lb, yb, pb_path, gb = load(base_run, "heldout")
    order_b = np.argsort(pb_path)
    correct_b = (lb.argmax(1) == yb)[order_b]

    def record(logits, y, groups, run):
        p = softmax(logits)
        m = classification_metrics(y, p, DISEASE_CLASSES)
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "n": m["n"],
                "ci95_macro_f1": bootstrap_ci(y, p, macro_f1, groups=groups),
                "best_epoch": json.loads((run / "summary.json").read_text())["best_epoch"],
                "img_size": cfg.get("img_size"), "pretrained": cfg.get("pretrained"),
                "augmentation": cfg.get("augmentation")}

    out = {"baseline": {**record(lb, yb, gb, base_run), "run": f"cv/{arch}/fold{fold}"}}
    for variant in sorted(p.name for p in abl.iterdir() if p.is_dir()):
        run = abl / variant / f"fold{fold}"
        if not (run / "summary.json").exists():
            continue
        lv, yv, pv_path, gv = load(run, "heldout")
        order_v = np.argsort(pv_path)
        assert np.array_equal(pv_path[order_v], pb_path[order_b]), f"{variant}: held-out images differ from baseline"
        rec = {**record(lv, yv, gv, run), "run": f"ablation/{arch}/{variant}/fold{fold}"}
        rec["delta_macro_f1_vs_baseline"] = rec["macro_f1"] - out["baseline"]["macro_f1"]
        rec["mcnemar_vs_baseline"] = mcnemar_exact((lv.argmax(1) == yv)[order_v], correct_b)
        out[variant] = rec
    return out if len(out) > 1 else {}


# ----------------------------------------------------------------------------- latency
def load_latency():
    p = RESULTS / "latency.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ----------------------------------------------------------------------------- tables & figures
def write_tables(cv, tests, selected, finals, ablation, latency):
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = ["| Model | Accuracy | Macro-F1 | Balanced acc. | Macro AUC (OvR) | Params (M) | CPU latency (ms) |",
             "|---|---|---|---|---|---|---|"]
    for arch, r in cv.items():
        lat = latency.get(arch, {})
        lines.append(f"| {arch}{' **(selected)**' if arch == selected else ''} | "
                     + " | ".join(fmt(r[k]["mean"], r[k]["sd"]) for k in
                                  ("accuracy", "macro_f1", "balanced_accuracy", "macro_auc_ovr"))
                     + f" | {lat.get('params_m', float('nan')):.1f} | {lat.get('median_ms', float('nan')):.1f} |")
    (TABLES / "cv_comparison.md").write_text(
        "5-fold group cross-validation (mean ± SD over folds). Source: results/disease/summary.json\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")

    if tests:
        t = ["| Pair | McNemar b/c | McNemar p (Holm) | Wilcoxon p (Holm) |", "|---|---|---|---|"]
        for k, v in tests["mcnemar_pooled_oof"].items():
            w = tests["wilcoxon_fold_macro_f1"][k]
            t.append(f"| {k} | {v['a_right_b_wrong']}/{v['a_wrong_b_right']} | {v['p_value']:.4f} ({v['p_holm']:.4f}) | "
                     f"{w['p_value']:.4f} ({w['p_holm']:.4f}) |")
        (TABLES / "significance.md").write_text(
            "Paired tests on CV results. McNemar: pooled out-of-fold predictions (b = first right/second wrong, "
            "c = reverse). Wilcoxon: per-fold macro-F1, n = 5 (minimum attainable p = 0.0625).\n\n"
            + "\n".join(t) + "\n", encoding="utf-8")

    if cv and selected:
        pc = cv[selected]["pooled_oof"]["per_class"]
        t = ["| Class | Precision | Recall | F1 | AUC | Support |", "|---|---|---|---|---|---|"]
        for c in DISEASE_CLASSES:
            v = pc[c]
            t.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['auc_ovr']:.3f} | {v['support']} |")
        (TABLES / "per_class_cv_selected.md").write_text(
            f"Per-class metrics of {selected}, pooled out-of-fold predictions.\n\n" + "\n".join(t) + "\n",
            encoding="utf-8")

    t = ["| Model | Test acc. [95% CI] | Test macro-F1 [95% CI] | T fitted → deployed | ECE test / external "
         "| External macro-F1 (4 cls) [95% CI] | AUROC near / far OOD | FPR@95 near / far | Flagged low-confidence: "
         "test / external |", "|---|---|---|---|---|---|---|---|---|"]
    for arch, r in finals.items():
        if not r:
            continue
        te, ex, ood = r["test"], r["external"], r["ood"]
        tfit = f"{r['temperature_fitted']:.2f}{'' if r['temperature_identifiable'] else ' (n.i.)'}"
        t.append(f"| {arch} | {te['accuracy']:.3f} [{te['ci95_accuracy'][0]:.3f}, {te['ci95_accuracy'][1]:.3f}] | "
                 f"{te['macro_f1']:.3f} [{te['ci95_macro_f1'][0]:.3f}, {te['ci95_macro_f1'][1]:.3f}] | "
                 f"{tfit} → {r['temperature']:.2f} | "
                 f"{r['calibration_test']['ece_after']:.3f} / {r['calibration_external']['ece_deployed_T']:.3f} | "
                 f"{ex['macro_f1']:.3f} [{ex['ci95_macro_f1'][0]:.3f}, {ex['ci95_macro_f1'][1]:.3f}] | "
                 f"{ood['ood_near']['auroc']:.3f} / {ood['ood_far']['auroc']:.3f} | "
                 f"{ood['ood_near']['fpr_at_95_tpr']:.2f} / {ood['ood_far']['fpr_at_95_tpr']:.2f} | "
                 f"{r['test_rejected_by_ood_threshold']:.3f} / {ex['rejected_by_ood_threshold']:.3f} |")
    (TABLES / "final_models.md").write_text(
        "Final models (train split, early stopping on val), evaluated once. CIs: leaf-level bootstrap. "
        "T: temperature fitted on val; '(n.i.)' = not identifiable (val perfectly separated, or the fit on the "
        "search bound), so T = 1 is deployed (post-hoc rule, disclosed). Low-confidence flag: top probability below the value that "
        "keeps 95% of val images; flagged images get a 'retake photo' message.\n\n"
        + "\n".join(t) + "\n", encoding="utf-8")

    if selected and finals.get(selected):
        ex = finals[selected]["external"]["per_class"]
        t = ["| Class | Recall | Precision | Support |", "|---|---|---|---|"]
        for c in DISEASE_CLASSES:
            v = ex[c]
            recall = "—" if v["recall"] is None else f"{v['recall']:.3f}"
            t.append(f"| {c} | {recall} | {v['precision']:.3f} | {v['support']} |")
        (TABLES / "external_selected.md").write_text(
            f"External test set, {selected} (temperature-scaled). Healthy is absent from this set.\n\n"
            + "\n".join(t) + "\n", encoding="utf-8")

    if ablation:
        t = ["| Variant | Accuracy | Macro-F1 [95% CI] | Δ macro-F1 | McNemar p | Best epoch |",
             "|---|---|---|---|---|---|"]
        for v, r in ablation.items():
            lo, hi = r["ci95_macro_f1"]
            d = "—" if v == "baseline" else f"{r['delta_macro_f1_vs_baseline']:+.3f}"
            p = "—" if v == "baseline" else f"{r['mcnemar_vs_baseline']['p_value']:.3g}"
            t.append(f"| {v} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}] | {d} | {p} | "
                     f"{r['best_epoch']} |")
        n = ablation["baseline"]["n"]
        (TABLES / "ablation.md").write_text(
            f"Single-run ablations of the EfficientNet-B3 recipe on CV fold 0 (held-out n = {n} images; baseline = "
            "the CV fold-0 run). One run per variant: no seed variance is estimated.\n\n"
            + "\n".join(t) + "\n", encoding="utf-8")


def write_figures(cv, selected, finals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)

    def cm_plot(cm, title, path, classes=DISEASE_CLASSES):
        cm = np.asarray(cm, float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_xlabel("Predicted"), ax.set_ylabel("True"), ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    if selected and selected in cv:
        cm_plot(cv[selected]["pooled_oof"]["confusion_matrix"], f"{selected}: 5-fold CV (pooled)",
                FIGURES / "cm_cv_selected.png")
    if selected and finals.get(selected):
        cm_plot(finals[selected]["test"]["confusion_matrix"], f"{selected}: test split", FIGURES / "cm_test_selected.png")
        cm_plot(finals[selected]["external"]["confusion_matrix"], f"{selected}: external set",
                FIGURES / "cm_external_selected.png")
        run = RESULTS / "final" / selected
        lt, yt, _, _ = load(run, "test")
        T = finals[selected]["temperature"]
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        for p, label in ((softmax(lt), "before"), (softmax(lt, T), f"after (T={T:.2f})")):
            conf, correct = p.max(1), p.argmax(1) == yt
            edges = np.linspace(0, 1, 11)
            xs, accs = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 3:
                    xs.append(conf[m].mean()), accs.append(correct[m].mean())
            ax.plot(xs, accs, "o-", label=label)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("Confidence"), ax.set_ylabel("Accuracy"), ax.set_title(f"Reliability, test ({selected})", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "reliability_test_selected.png", dpi=150)
        plt.close(fig)
    if cv:
        archs = list(cv)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        means = [cv[a]["macro_f1"]["mean"] for a in archs]
        sds = [cv[a]["macro_f1"]["sd"] for a in archs]
        ax.bar(range(len(archs)), means, yerr=sds, capsize=4, color="#4c8c4a")
        for k, a in enumerate(archs):
            ax.scatter([k] * 5, cv[a]["macro_f1"]["per_fold"], color="black", s=8, zorder=3)
        ax.set_xticks(range(len(archs)), archs, rotation=20, fontsize=8)
        ax.set_ylabel("Macro-F1 (5-fold group CV)")
        ax.set_ylim(max(0.0, min(min(cv[a]["macro_f1"]["per_fold"]) for a in archs) - 0.05), 1.0)
        fig.tight_layout()
        fig.savefig(FIGURES / "cv_macro_f1.png", dpi=150)
        plt.close(fig)


def main():
    cv, tests, selected = analyse_cv()
    finals = {arch: analyse_final(arch) for arch in ARCHS}
    finals = {k: v for k, v in finals.items() if v}
    ablation = analyse_ablation()
    latency = load_latency()
    summary = {"git": git_info(), "selection_rule": "highest mean CV macro-F1", "selected": selected,
               "study_archs": list(ARCHS), "deferred_not_run": list(DEFERRED_ARCHS),
               "cv": cv, "cv_tests": tests, "final": finals, "ablation": ablation, "latency": latency,
               "complete": {"cv": sorted(cv), "final": sorted(finals), "ablation": sorted(ablation)}}
    write_json(RESULTS / "summary.json", summary)
    write_tables(cv, tests, selected, finals, ablation, latency)
    write_figures(cv, selected, finals)
    print(f"selected: {selected}; complete: {summary['complete']}")
    for name in sorted(TABLES.glob("*.md")):
        print(f"\n## {name.name}\n" + name.read_text(encoding="utf-8"))
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")      # tables contain non-ASCII (e.g. the delta sign)
    main()
