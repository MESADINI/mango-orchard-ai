#!/usr/bin/env python3
"""
Deploys the selected disease model (results/disease/summary.json -> "selected", chosen by mean CV macro-F1) to
ml/disease_detection/weights/disease_classifier.pth, embedding the temperature (fitted on val) and the OOD
rejection threshold (keeps 95% of val images) so that inference can say "unsure - retake the photo".

    python ml/disease_detection/deploy.py [--arch ARCH]

Writes <checkpoint>.sha256 and results/disease/deployed.json (provenance: source run, git, hashes).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.disease_detection.train import DEFAULT_SAVE_DIR, CHECKPOINT_NAME, sha256_file  # noqa: E402
from ml.utils.experiment import git_info, write_json  # noqa: E402

RESULTS = REPO_ROOT / "results" / "disease"


def deploy(arch=None, dest: Path = DEFAULT_SAVE_DIR / CHECKPOINT_NAME) -> dict:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    arch = arch or summary["selected"]
    final = summary["final"][arch]
    src = RESULTS / "final" / arch / "best.pth"
    src_sha = sha256_file(src)
    if src_sha != final["summary"]["checkpoint_sha256"]:
        raise RuntimeError(f"{src} does not match the SHA-256 recorded at training time")
    ckpt = torch.load(src, map_location="cpu")
    ckpt.update({"temperature": final["temperature"], "ood_threshold": final["ood_threshold"],
                 "source_run": str(src.relative_to(REPO_ROOT).as_posix()), "source_sha256": src_sha,
                 "selection": summary["selection_rule"]})
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dest)
    digest = sha256_file(dest)
    dest.with_suffix(dest.suffix + ".sha256").write_text(f"{digest}  {dest.name}\n", encoding="ascii")
    record = {"arch": arch, "deployed_to": str(dest.relative_to(REPO_ROOT).as_posix()), "sha256": digest,
              "source_run": ckpt["source_run"], "source_sha256": src_sha, "temperature": final["temperature"],
              "ood_threshold": final["ood_threshold"], "selection": summary["selection_rule"],
              "selected_by_rule": summary["selected"], "git": git_info()}
    write_json(RESULTS / "deployed.json", record)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default=None, help="override the selected architecture (recorded)")
    deploy(ap.parse_args().arch)
