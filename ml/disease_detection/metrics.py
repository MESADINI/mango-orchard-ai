"""
Evaluation metrics for the disease classifier (pure numpy/scipy; unit-tested).

All functions take integer labels y [N] and either predictions [N] or probabilities [N, C].
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


def classification_metrics(y: np.ndarray, probs: np.ndarray, classes: Sequence[str]) -> Dict:
    """Accuracy, macro-F1, balanced accuracy, per-class P/R/F1/support, confusion matrix, macro one-vs-rest AUC.
    Classes absent from y get recall/AUC = None and are left out of the macro averages."""
    from sklearn.metrics import confusion_matrix, roc_auc_score

    y = np.asarray(y)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(1)
    labels = list(range(len(classes)))
    cm = confusion_matrix(y, pred, labels=labels)
    per_class, f1s, recalls, aucs = {}, [], [], []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if support else None
        auc = None
        if support and support < len(y):
            auc = float(roc_auc_score(y == i, probs[:, i]))
            aucs.append(auc)
        if support:
            f1s.append(f1)
            recalls.append(recall)
        per_class[c] = {"precision": float(precision), "recall": None if recall is None else float(recall),
                        "f1": None if f1 is None else float(f1), "support": support, "auc_ovr": auc}
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_auc_ovr": float(np.mean(aucs)) if aucs else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes_evaluated": [c for c, v in per_class.items() if v["support"]],
    }


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(1)
    correct = probs.argmax(1) == np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


T_BOUNDS = (0.05, 20.0)


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds=T_BOUNDS) -> float:
    """Temperature T > 0 minimising validation NLL (Guo et al., 2017); bounded scalar search over log T."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda lt: nll(logits, y, float(np.exp(lt))), bounds=(np.log(bounds[0]), np.log(bounds[1])),
                          method="bounded")
    return float(np.exp(res.x))


def temperature_at_bound(T: float, bounds=T_BOUNDS, rel_tol: float = 0.02) -> bool:
    """True when the fitted temperature sits on a search bound."""
    return T <= bounds[0] * (1 + rel_tol) or T >= bounds[1] * (1 - rel_tol)


def temperature_identifiable(logits: np.ndarray, y: np.ndarray, T: float, bounds=T_BOUNDS) -> bool:
    """False when the fit carries no information: if every validation prediction is correct, NLL decreases
    monotonically as T -> 0 (the optimiser stops at the bound or wherever the NLL becomes numerically flat), and a
    fit on the search bound is not an interior optimum either."""
    perfectly_separated = bool((np.asarray(logits).argmax(1) == np.asarray(y)).all())
    return not perfectly_separated and not temperature_at_bound(T, bounds)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict:
    """Exact (binomial) McNemar test on paired correctness vectors."""
    from scipy.stats import binomtest

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())   # A right, B wrong
    n10 = int((~a & b).sum())   # A wrong, B right
    p = 1.0 if n01 + n10 == 0 else float(binomtest(n01, n01 + n10, 0.5).pvalue)
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


def wilcoxon_paired(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict:
    """Two-sided exact Wilcoxon signed-rank test; also returns the smallest attainable p for this n."""
    from scipy.stats import wilcoxon

    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    n = len(a)
    min_p = 2.0 / 2 ** n
    if np.allclose(a, b):
        return {"n": n, "statistic": 0.0, "p_value": 1.0, "min_attainable_p": min_p}
    res = wilcoxon(a, b, alternative="two-sided", method="exact" if n <= 25 else "auto")
    return {"n": n, "statistic": float(res.statistic), "p_value": float(res.pvalue), "min_attainable_p": min_p}


def holm_adjust(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def ood_metrics(scores_in: np.ndarray, scores_out: np.ndarray, threshold: Optional[float] = None) -> Dict:
    """Higher score = more in-distribution. AUROC (in vs out), FPR of OOD at the threshold keeping 95% of
    in-distribution samples, and, if given, the in-distribution acceptance / OOD rejection rate at `threshold`."""
    from sklearn.metrics import roc_auc_score

    s_in, s_out = np.asarray(scores_in, float), np.asarray(scores_out, float)
    y = np.r_[np.ones(len(s_in)), np.zeros(len(s_out))]
    t95 = float(np.quantile(s_in, 0.05))
    out = {
        "n_in": int(len(s_in)), "n_out": int(len(s_out)),
        "auroc": float(roc_auc_score(y, np.r_[s_in, s_out])),
        "fpr_at_95_tpr": float((s_out >= t95).mean()),
    }
    if threshold is not None:
        out.update({"threshold": float(threshold), "in_accepted": float((s_in >= threshold).mean()),
                    "out_rejected": float((s_out < threshold).mean())})
    return out


def bootstrap_ci(y: np.ndarray, probs: np.ndarray, metric, groups: Optional[np.ndarray] = None,
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> List[float]:
    """Percentile CI of metric(y, probs). With `groups`, resamples whole groups (leaves), not images."""
    rng = np.random.default_rng(seed)
    y, probs = np.asarray(y), np.asarray(probs)
    if groups is None:
        groups = np.arange(len(y))
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        vals.append(metric(y[idx], probs[idx]))
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]
