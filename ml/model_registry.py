#!/usr/bin/env python3
"""
Model registry (Phase 6): one committed record per deployed model, so every prediction can be traced to the exact
weights, classes and thresholds that produced it.

    python ml/model_registry.py build      -> ml/model_registry.json (from results/{disease,pest}/deployed.json)
    python ml/model_registry.py verify     -> checks that the installed weights match the registry (exit 1 if not)

Each entry: name, task, version (first 12 hex digits of the weights' SHA-256), sha256, path, classes, calibration
(temperature, low-confidence threshold), host_verified flags (pest), source run, git commit, selection rule, and the
headline metrics copied from results/. Weights themselves are not committed (they are built by the training scripts).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGISTRY = REPO_ROOT / "ml" / "model_registry.json"
DISEASE_DEPLOYED = REPO_ROOT / "results" / "disease" / "deployed.json"
DISEASE_EVAL = REPO_ROOT / "results" / "disease" / "deployed_eval.json"
PEST_DEPLOYED = REPO_ROOT / "results" / "pest" / "deployed.json"
PEST_WEIGHTS = "ml/pest_detection/weights/pest_detector.pt"


def _load(path: Path) -> Optional[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _sha256(path: Path) -> str:
    from ml.datasets.fetch import sha256_file
    return sha256_file(Path(path))


def disease_entry(deployed: dict, evaluation: Optional[dict] = None) -> dict:
    from ml.disease_detection.inference import DISEASE_CLASSES
    return {
        "name": "disease_classifier", "task": "leaf disease classification", "arch": deployed["arch"],
        "version": deployed["sha256"][:12], "sha256": deployed["sha256"], "path": deployed["deployed_to"],
        "classes": list(DISEASE_CLASSES),
        "calibration": {"temperature": deployed["temperature"], "low_confidence_threshold": deployed["ood_threshold"],
                        "low_confidence_rule": "top probability below the value that keeps 95% of validation images "
                                               "-> 'retake photo', disease excluded from the OHI"},
        "source_run": deployed["source_run"], "source_sha256": deployed["source_sha256"],
        "git_commit": deployed.get("git", {}).get("commit"), "selection": deployed.get("selection"),
        "metrics": evaluation or {},
    }


def pest_entry(deployed: dict) -> dict:
    return {
        "name": "pest_detector", "task": "pest detection (boxes)", "arch": deployed["model"],
        "version": deployed["sha256"][:12], "sha256": deployed["sha256"], "path": PEST_WEIGHTS,
        "classes": deployed["classes"], "host_verified": deployed["host_verified"],
        "source_run": deployed["source_checkpoint"], "selection": deployed.get("selection"),
        "metrics": {"val_map50_95": deployed.get("val_map50_95"), "test_map50_95": deployed.get("test_map50_95")},
        "notes": "Classes learned from non-mango sources (ml/datasets/pest_source_review.json); detections are shown "
                 "as unverified and the pest component is excluded from the OHI while no class is host-verified.",
    }


def build(disease_deployed: Path = DISEASE_DEPLOYED, disease_eval: Path = DISEASE_EVAL,
          pest_deployed: Path = PEST_DEPLOYED, out: Path = REGISTRY) -> dict:
    models = []
    d = _load(disease_deployed)
    if d:
        ev = _load(disease_eval) or {}
        metrics = {k: ev[k] for k in ("test", "external") if k in ev}
        models.append(disease_entry(d, {s: {m: v.get(m) for m in ("accuracy", "macro_f1", "low_confidence_fraction", "n")}
                                        for s, v in metrics.items()}))
    p = _load(pest_deployed)
    if p:
        models.append(pest_entry(p))
    reg = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": models}
    Path(out).write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    return reg


def status(registry: Path = REGISTRY, root: Path = REPO_ROOT) -> Dict[str, dict]:
    """Per registered model: installed? does its SHA-256 match the registry? ("ok" / "missing" / "sha_mismatch")."""
    reg = _load(registry) or {"models": []}
    out = {}
    for m in reg["models"]:
        path = Path(root) / m["path"]
        state = "missing" if not path.exists() else ("ok" if _sha256(path) == m["sha256"] else "sha_mismatch")
        out[m["name"]] = {"version": m["version"], "arch": m["arch"], "status": state, "classes": m["classes"],
                          **({"calibration": m["calibration"]} if "calibration" in m else {}),
                          **({"host_verified": m["host_verified"]} if "host_verified" in m else {})}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["build", "verify"])
    a = ap.parse_args()
    if a.what == "build":
        print(json.dumps(build(), indent=2))
    else:
        st = status()
        print(json.dumps(st, indent=2))
        sys.exit(0 if st and all(v["status"] == "ok" for v in st.values()) else 1)
