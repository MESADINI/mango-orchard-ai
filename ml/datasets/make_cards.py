#!/usr/bin/env python3
"""
Fills the auto-generated sections of the dataset cards (docs/datasets/*.md) from results/data/*.json,
so numbers in the cards can never drift from the evidence. A section looks like:

    <!-- AUTO:disease_split_table -->
    ...generated...
    <!-- /AUTO -->
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "results" / "data"
CARDS = REPO_ROOT / "docs" / "datasets"


def _load(name):
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def split_class_table(per_split_class: dict, classes: list) -> str:
    splits = ["train", "val", "test"]
    lines = ["| Class | " + " | ".join(splits) + " | Total |", "|---|" + "---|" * (len(splits) + 1)]
    totals = {s: 0 for s in splits}
    for c in classes:
        row = [per_split_class.get(s, {}).get(c, 0) for s in splits]
        for s, v in zip(splits, row):
            totals[s] += v
        lines.append(f"| {c} | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |")
    lines.append("| **Total** | " + " | ".join(str(totals[s]) for s in splits) + f" | {sum(totals.values())} |")
    return "\n".join(lines)


def kv_table(pairs) -> str:
    return "\n".join(["| Quantity | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in pairs])


def sections() -> dict:
    out = {}
    dl, ds, ext = _load("mangoleafbd_download.json"), _load("disease_dataset.json"), _load("external_testset.json")
    if dl:
        out["mangoleafbd_source_table"] = kv_table([
            ("Images in release", dl["n_images"]),
            ("Unique by SHA-256", dl["n_unique_sha256"]),
            ("Matched to the authors' Mendeley SHA-256 list", f"{dl['matched_to_mendeley_sha256']} / {dl['mendeley_files']}"),
            ("Each Mendeley folder maps to exactly one class", dl["every_mendeley_folder_maps_to_one_class"]),
            ("Images per source class", ", ".join(f"{k}: {v}" for k, v in dl["per_class"].items())),
            ("Most common sizes (W x H)", ", ".join(f"{k}: {v}" for k, v in dl["image_sizes_top5"].items())),
        ])
    if ds:
        d = ds["dedup"]
        out["disease_dedup_table"] = kv_table([
            ("pHash threshold (dihedral min Hamming)", d["phash_threshold"]),
            ("DINOv2 candidates per image (retrieval only)", d["candidate_k"]),
            ("SIFT + RANSAC inlier threshold (link = same leaf)", d["inlier_threshold"]),
            ("Candidate pairs verified / linked", f"{d['n_candidate_pairs']} / {d['n_verified_edges']}"),
            ("pHash-identical links", d["n_phash_edges"]),
            ("Exact duplicates removed", d["n_exact_duplicates_removed"]),
            ("Unique images (all 8 classes)", d["n_unique_images"]),
            ("Groups (= estimated distinct leaves, all 8 classes)", d["n_groups_all_classes"]),
            ("Images in groups of size > 1", d["n_images_in_multi_image_groups"]),
            ("Cross-class conflict groups (excluded)", d["n_cross_class_conflict_groups"]),
            ("Excluded images", ", ".join(f"{k}: {v}" for k, v in ds["excluded_images"].items())),
            ("Images in final dataset", ds["n_images"]),
            ("Groups in final dataset", ds["n_groups"]),
            ("Group overlap between splits", ", ".join(f"{k}: {v}" for k, v in ds["group_overlap_between_splits"].items())),
        ])
        out["disease_split_table"] = split_class_table(ds["images_per_split_class"], ds["classes"])
    if ext:
        out["external_table"] = kv_table([
            ("Candidates (processed, known classes, unique)", ext["n_candidates"]),
            ("At MangoLeafBD pixel sizes (excluded, probable derivatives)", ext["n_mangoleafbd_format"]),
            ("... of which confirmed as MangoLeafBD copies by the leakage guard", ext["n_mangoleafbd_format_confirmed_by_guard"]),
            ("At native camera resolution", ext["n_native_format"]),
            ("... of which flagged by the leakage guard (excluded)", ext["n_native_format_flagged_by_guard"]),
            ("Candidates per class and format", ", ".join(f"{k.replace('|', ' / ')}: {v}" for k, v in ext["candidates_per_class_and_format"].items())),
            ("Images in external test set", ext["n_images"]),
            ("Per class", ", ".join(f"{k}: {v}" for k, v in ext["per_class"].items())),
            ("Known classes missing", ", ".join(ext["classes_missing"]) or "none"),
            ("Skipped", ", ".join(f"{k}: {v}" for k, v in ext["skipped"].items())),
        ])
    pest = _load("pest_dataset.json")
    if pest:
        per_source = [(f"`{slug}` ({s.get('status', '')})",
                       f"{s['n_images_in_export']} exported; {s.get('exact_duplicates_removed', 0)} exact dups and "
                       f"{s.get('images_dropped_only_other_classes', 0)} only-dropped-class images removed; "
                       f"boxes kept {s.get('label_stats', {}).get('kept', 0)}, dropped "
                       f"{s.get('label_stats', {}).get('dropped_other_class', 0)}")
                      for slug, s in pest["sources"].items()]
        out["pest_table"] = kv_table(per_source + [
            ("Included non-mango (flagged proxy) sources", pest["include_non_mango"]),
            ("Excluded sources", ", ".join(f"{k}: {v}" for k, v in pest["excluded_sources"].items()) or "none"),
            ("Images / groups", f"{pest['n_images']} / {pest['n_groups']}"),
            ("Images without boxes (true background)", pest["images_without_boxes"]),
            ("Group overlap between splits", ", ".join(f"{k}: {v}" for k, v in pest["group_overlap_between_splits"].items())),
        ]) + "\n\n**Images per split**\n\n" + split_class_table(pest["images_per_split_class"], pest["classes"]) \
            + "\n\n**Box instances per split**\n\n" + split_class_table(pest["instances_per_split_class"], pest["classes"])
    return out


def fill(text: str, generated: dict) -> str:
    def repl(m):
        name = m.group(1)
        body = generated.get(name, "_Not yet generated — run the dataset build scripts._")
        return f"<!-- AUTO:{name} -->\n{body}\n<!-- /AUTO -->"
    return re.sub(r"<!-- AUTO:(\w+) -->.*?<!-- /AUTO -->", repl, text, flags=re.DOTALL)


def main():
    generated = sections()
    for card in sorted(CARDS.glob("*.md")):
        text = card.read_text(encoding="utf-8")
        new = fill(text, generated)
        if new != text:
            card.write_text(new, encoding="utf-8")
            print(f"updated {card.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
