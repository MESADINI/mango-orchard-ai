#!/usr/bin/env python3
"""
CPU inference latency and model size per architecture (deployment is CPU-only in the Docker image).

    python ml/disease_detection/latency.py [--threads 4] [--runs 50]

Batch 1, 224x224, torch.inference_mode, fp32, after 10 warm-up runs; reports median and p90 over `runs`.
Model size = parameters and the size of the trained checkpoint from results/disease/final/<arch>/best.pth if
present (otherwise of a freshly saved state_dict). Output: results/disease/latency.json
"""

import argparse
import io
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.train import STUDY_ARCHS, create_model  # noqa: E402
from ml.utils.experiment import collect_env, write_json  # noqa: E402

OUT = REPO_ROOT / "results" / "disease" / "latency.json"


def measure(arch: str, threads: int, runs: int, img_size: int = 224) -> dict:
    torch.set_num_threads(threads)
    model = create_model(arch, pretrained=False).eval()
    x = torch.randn(1, 3, img_size, img_size)
    times = []
    with torch.inference_mode():
        for i in range(10 + runs):
            t = time.perf_counter()
            model(x)
            if i >= 10:
                times.append((time.perf_counter() - t) * 1000)
    ckpt = REPO_ROOT / "results" / "disease" / "final" / arch / "best.pth"
    if ckpt.exists():
        size = ckpt.stat().st_size
    else:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        size = buf.tell()
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6, "checkpoint_mb": size / 2 ** 20,
            "checkpoint_from_trained_run": ckpt.exists()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()
    res = {arch: measure(arch, a.threads, a.runs) for arch in STUDY_ARCHS}
    res["_setup"] = {"threads": a.threads, "runs": a.runs, "batch": 1, "img_size": 224, "dtype": "float32",
                     "cpu": platform.processor(), "env": collect_env()}
    write_json(OUT, res)
    for k, v in res.items():
        if not k.startswith("_"):
            print(f"{k}: {v['median_ms']:.1f} ms (p90 {v['p90_ms']:.1f}), {v['params_m']:.1f} M params, {v['checkpoint_mb']:.1f} MB")
