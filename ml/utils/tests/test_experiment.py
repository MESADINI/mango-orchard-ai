import json
import random

import pytest
import torch

from ml.utils.experiment import collect_env, set_seed, start_experiment


def test_set_seed_makes_rngs_reproducible():
    set_seed(123)
    a = (random.random(), torch.rand(3).tolist())
    set_seed(123)
    b = (random.random(), torch.rand(3).tolist())
    assert a == b


def test_collect_env_records_git_and_versions():
    env = collect_env()
    assert env["git"]["commit"]  # the test suite runs inside the repository
    assert env["packages"]["torch"].split("+")[0] == torch.__version__.split("+")[0]
    assert "cuda_available" in env["hardware"]


def test_start_experiment_writes_config_and_env(tmp_path):
    out = start_experiment("unit_exp", {"seed": 7, "lr": 0.1}, results_dir=tmp_path)
    assert json.loads((out / "config.json").read_text())["lr"] == 0.1
    assert "git" in json.loads((out / "env.json").read_text())


def test_start_experiment_rejects_bad_ids(tmp_path):
    with pytest.raises(ValueError):
        start_experiment("bad:id", {}, results_dir=tmp_path)
