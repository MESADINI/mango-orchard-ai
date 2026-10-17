import json

import numpy as np

from ml.disease_detection import analysis
from ml.disease_detection.inference import DISEASE_CLASSES

C = len(DISEASE_CLASSES)


def _write_run(d, sets, rng, skill):
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"best_epoch": 3}))
    for name, (y, paths) in sets.items():
        logits = rng.normal(size=(len(y), C))
        known = y >= 0
        logits[np.where(known)[0], y[known]] += skill
        if not known.all():
            logits[~known] *= 0.2          # OOD images: flatter outputs
        np.savez(d / f"logits_{name}.npz", logits=logits, y=y, path=np.array(paths),
                 group=np.array([str(i // 2) for i in range(len(y))]))


def test_full_analysis_on_synthetic_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    monkeypatch.setattr(analysis, "RESULTS", tmp_path)
    monkeypatch.setattr(analysis, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(analysis, "FIGURES", tmp_path / "figures")
    archs = list(analysis.ARCHS)[:2]
    skills = {archs[0]: 4.0, archs[1]: 1.0}
    for arch in archs:
        for k in range(5):
            y = rng.integers(0, C, 40)
            _write_run(tmp_path / "cv" / arch / f"fold{k}",
                       {"heldout": (y, [f"f{k}_{i}.jpg" for i in range(40)])}, np.random.default_rng(k), skills[arch])
        y_ood = -np.ones(30, int)
        _write_run(tmp_path / "final" / arch, {
            "val": (rng.integers(0, C, 60), [f"v{i}" for i in range(60)]),
            "test": (rng.integers(0, C, 60), [f"t{i}" for i in range(60)]),
            "external": (rng.integers(0, C - 1, 40), [f"e{i}" for i in range(40)]),
            "ood_near": (y_ood, [f"n{i}" for i in range(30)]),
            "ood_far": (y_ood, [f"o{i}" for i in range(30)]),
        }, rng, skills[arch])
    z = np.load(tmp_path / "cv" / "efficientnet_b3" / "fold0" / "logits_heldout.npz")   # baseline held-out images
    held0 = (z["y"], list(z["path"]))
    for v, skill in (("no_pretraining", 0.5), ("img_size_256", 4.0)):
        _write_run(tmp_path / "ablation" / "efficientnet_b3" / v / "fold0", {"heldout": held0},
                   np.random.default_rng(7), skill)

    s = analysis.main()
    assert s["selected"] == archs[0]                       # higher-skill model wins on CV macro-F1
    assert s["cv"][archs[0]]["macro_f1"]["mean"] > s["cv"][archs[1]]["macro_f1"]["mean"]
    pair = f"{archs[0]} vs {archs[1]}"
    assert s["cv_tests"]["mcnemar_pooled_oof"][pair]["p_value"] < 0.05
    assert s["cv_tests"]["wilcoxon_fold_macro_f1"][pair]["min_attainable_p"] == 0.0625
    f = s["final"][archs[0]]
    assert f["temperature"] > 0 and 0 <= f["calibration_test"]["ece_after"] <= 1
    assert f["external"]["per_class"]["Healthy"]["support"] == 0
    assert f["ood"]["ood_near"]["auroc"] > 0.5
    assert set(s["ablation"]) == {"baseline", "no_pretraining", "img_size_256"}
    assert s["ablation"]["baseline"]["run"] == f"cv/{archs[0]}/fold0"
    assert s["ablation"]["no_pretraining"]["delta_macro_f1_vs_baseline"] < 0
    assert s["ablation"]["no_pretraining"]["mcnemar_vs_baseline"]["p_value"] < 0.05
    assert s["deferred_not_run"] == ["convnext_tiny", "swin_tiny"]
    for name in ("cv_comparison.md", "significance.md", "final_models.md", "ablation.md", "external_selected.md"):
        assert (tmp_path / "tables" / name).exists()
    assert (tmp_path / "figures" / "cm_test_selected.png").exists()
