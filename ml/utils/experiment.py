"""
Experiment bookkeeping shared by every training / evaluation script.

Each experiment writes to results/<exp_id>/:
    config.json  - the exact configuration used
    env.json     - git commit, dirty flag, Python/package versions, hardware
so that every reported number can be traced to the code and environment that produced it.
"""

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

TRACKED_PACKAGES = [
    "torch", "torchvision", "timm", "ultralytics", "numpy", "Pillow", "opencv-python-headless",
    "scikit-learn", "scipy", "pandas", "matplotlib", "ImageHash", "kaggle",
]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seeds Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_info(repo: Path = REPO_ROOT) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions(packages=TRACKED_PACKAGES) -> Dict[str, Optional[str]]:
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": False,
        "gpu": None,
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        info["nvidia_smi"] = out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info["nvidia_smi"] = None
    try:
        import torch
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def collect_env() -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "git": git_info(),
        "packages": package_versions(),
        "hardware": hardware_info(),
    }


def write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def start_experiment(exp_id: str, config: Dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Creates results/<exp_id>/ with config.json and env.json, seeds RNGs, and returns the directory."""
    if not exp_id or any(c in exp_id for c in "\\:*?\"<>|"):
        raise ValueError(f"Invalid experiment id: {exp_id!r}")
    out_dir = Path(results_dir) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if "seed" in config:
        set_seed(int(config["seed"]))
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "env.json", collect_env())
    return out_dir
