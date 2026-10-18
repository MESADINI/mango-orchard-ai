from ml.disease_detection import experiments
from ml.disease_detection.experiments import ABLATIONS, cv_partition
from ml.disease_detection.train import DEFAULT_CONFIG, STUDY_ARCHS


def _rows(n=200):
    return [{"path": f"{i}.jpg", "y": i % 5, "group": str(i // 4), "cv_fold": i % 5} for i in range(n)]


def test_cv_partition_rules_are_disjoint():
    rows = _rows() + [{"path": "big.jpg", "y": 0, "group": "big", "cv_fold": -2},
                      {"path": "t.jpg", "y": 0, "group": "t", "cv_fold": -1}]
    for k in range(5):
        train, val, held = cv_partition(rows, k)
        assert {r["cv_fold"] for r in held} == {k} and {r["cv_fold"] for r in val} == {(k + 1) % 5}
        names = [set(r["path"] for r in s) for s in (train, val, held)]
        assert not (names[0] & names[1]) and not (names[0] & names[2]) and not (names[1] & names[2])
        assert "big.jpg" in names[0] and all("t.jpg" not in s for s in names)


def test_ablations_change_one_thing_each():
    assert set(ABLATIONS) == {"no_pretraining", "basic_augmentation", "img_size_256"}
    for v in ABLATIONS.values():
        assert len(v) == 1 and all(DEFAULT_CONFIG[k] != val for k, val in v.items())


def test_ablations_train_on_the_cv_fold_partition(monkeypatch, tmp_path):
    rows = _rows()
    calls = []
    monkeypatch.setattr(experiments, "read_split", lambda: rows)
    monkeypatch.setattr(experiments, "RESULTS", tmp_path)
    monkeypatch.setattr(experiments, "run_training",
                        lambda arch, tr, va, out, ev, cfg: calls.append((arch, tr, va, out, ev, cfg)))
    experiments.run_ablation("efficientnet_b3", fold=0)
    assert len(calls) == 3
    train, val, held = cv_partition(rows, 0)
    for arch, tr, va, out, ev, cfg in calls:
        assert (tr, va, list(ev)) == (train, val, ["heldout"]) and ev["heldout"] == held
        assert out.name == "fold0" and cfg["cv_fold"] == 0 and cfg["early_stop_fold"] == 1
        assert "seed" not in cfg          # same seed as the baseline CV run (DEFAULT_CONFIG)


def test_deferred_architectures_are_not_in_the_study():
    assert STUDY_ARCHS == ("efficientnet_b3", "resnet50", "mobilenetv3_large")
