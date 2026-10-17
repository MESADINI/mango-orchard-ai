#!/usr/bin/env python3
"""
Cross-platform task runner. The Makefile delegates every target here, so
`make <target>` and `python tasks.py <target>` are equivalent (Windows has no make by default).

Targets:
    env            record the environment to results/environment/<timestamp>.json
    data           build the disease and pest datasets from scratch
    train-disease  train the disease classifier
    train-pest     train the pest detector
    eval           evaluate trained models on the held-out test split
    test           run the full pytest suite
    paper-tables   regenerate docs/RESULTS.md and the README results table from results/
    all            data -> train-disease -> train-pest -> eval -> test
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=REPO, check=True)


def env() -> None:
    sys.path.insert(0, str(REPO))
    from ml.utils.experiment import collect_env, write_json
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = write_json(REPO / "results" / "environment" / f"{stamp}.json", collect_env())
    print(f"Environment recorded to {path}")


def data() -> None:
    run(PY, "ml/datasets/download_mangoleafbd.py")
    run(PY, "ml/datasets/build_disease_dataset.py")
    run(PY, "ml/datasets/build_external_testset.py")
    run(PY, "ml/datasets/build_pest_dataset.py", "all", "--include-non-mango")  # flagged proxy sources
    run(PY, "ml/datasets/data_stats.py")


def train_disease() -> None:
    for what in ("cv", "final"):
        run(PY, "ml/disease_detection/experiments.py", what)
    run(PY, "ml/disease_detection/experiments.py", "ablation", "--arch", "efficientnet_b3")
    run(PY, "ml/disease_detection/latency.py")
    run(PY, "ml/disease_detection/analysis.py")
    run(PY, "ml/disease_detection/deploy.py")


def train_pest() -> None:
    run(PY, "ml/pest_detection/train.py", "train")       # YOLOv8n, YOLO11n, YOLOv8s -> results/pest/
    run(PY, "ml/pest_detection/train.py", "deploy")      # selected by val mAP50-95, SHA-verified copy
    run(PY, "ml/model_registry.py", "build")


def evaluate() -> None:
    run(PY, "ml/disease_detection/evaluate.py")


def test() -> None:
    run(PY, "-m", "pytest", "-q")


def paper_tables() -> None:
    run(PY, "ml/reporting/results_tables.py")


def all_targets() -> None:
    for step in (data, train_disease, train_pest, evaluate, test):
        step()


TARGETS = {
    "env": env,
    "data": data,
    "train-disease": train_disease,
    "train-pest": train_pest,
    "eval": evaluate,
    "test": test,
    "paper-tables": paper_tables,
    "all": all_targets,
}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        print(f"usage: python tasks.py {{{'|'.join(TARGETS)}}}")
        sys.exit(2)
    TARGETS[sys.argv[1]]()
