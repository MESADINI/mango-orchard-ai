#!/usr/bin/env python3
"""
API latency benchmark (Phase 6). In-process FastAPI TestClient, CPU only, installed real weights, weather stubbed
(so Open-Meteo's network time is excluded and runs are comparable), temporary database and media directory.

    python backend/benchmark_latency.py [--n 30] [--warmup 3]

Measures POST /api/analyze/ wall time per request for scan types disease, pest and dual (the full pipeline:
classification, Grad-CAM, severity, detection, OHI, recommendations, database write). Run on an idle machine.
Output: results/api/latency.{json,md}
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = REPO_ROOT / "results" / "api"


def main(n: int = 30, warmup: int = 3) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # CPU, like a typical deployment server
    tmp = Path(tempfile.mkdtemp(prefix="mango_bench_"))
    os.environ["DATABASE_PATH"] = str(tmp / "bench.db")
    os.environ["MEDIA_DIR"] = str(tmp / "media")

    import numpy as np
    from fastapi.testclient import TestClient
    from app.core.config import settings
    from app.main import app
    from app.services.pipeline_service import pipeline_service
    from ml.model_registry import status
    from ml.utils.experiment import collect_env, write_json
    from testing_support import SAMPLE_LEAF, SAMPLE_PEST, fake_weather

    results = {"n": n, "warmup": warmup, "device": "cpu", "weather": "stubbed (network excluded)",
               "models": status(), "scan_types": {}}
    with TestClient(app) as client:
        pipeline_service.get_pipeline().weather_fetcher = fake_weather
        token = client.post("/api/auth/login", json={"email": settings.DEMO_USER_EMAIL,
                                                     "password": settings.DEMO_USER_PASSWORD}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        for scan_type in ("disease", "pest", "dual"):
            times = []
            for i in range(warmup + n):
                with open(SAMPLE_LEAF, "rb") as lf, open(SAMPLE_PEST, "rb") as pf:
                    files = {"file": ("leaf.jpg", lf, "image/jpeg")}
                    if scan_type != "disease":
                        files["pest_file"] = ("pest.jpg", pf, "image/jpeg")
                    t0 = time.perf_counter()
                    res = client.post("/api/analyze/", files=files, data={"scan_type": scan_type}, headers=headers)
                    dt = (time.perf_counter() - t0) * 1000
                if res.status_code != 200:
                    raise RuntimeError(f"{scan_type}: HTTP {res.status_code} {res.text[:200]}")
                if i >= warmup:
                    times.append(dt)
            a = np.array(times)
            results["scan_types"][scan_type] = {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
                                                "mean_ms": float(a.mean()), "min_ms": float(a.min()),
                                                "max_ms": float(a.max())}
            print(scan_type, results["scan_types"][scan_type], flush=True)
    results["env"] = collect_env()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "latency.json", results)
    lines = [f"POST /api/analyze/ latency, CPU, in-process client, weather stubbed; {n} requests after {warmup} "
             "warm-up (results/api/latency.json).\n", "| Scan type | Median (ms) | p95 (ms) | Min | Max |",
             "|---|---|---|---|---|"]
    for k, v in results["scan_types"].items():
        lines.append(f"| {k} | {v['median_ms']:.0f} | {v['p95_ms']:.0f} | {v['min_ms']:.0f} | {v['max_ms']:.0f} |")
    (OUT / "latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    main(a.n, a.warmup)
