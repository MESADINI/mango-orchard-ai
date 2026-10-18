import csv
import json

import cv2
import numpy as np
from PIL import Image

from ml.severity import validate


def _leaf_image(tmp_path, name, lesion_r):
    img = np.full((300, 300, 3), 235, np.uint8)
    mask = np.zeros((300, 300), np.uint8)
    cv2.ellipse(mask, (150, 150), (120, 60), 0, 0, 360, validate.LEAF, -1)
    img[mask == validate.LEAF] = (60, 140, 50)
    if lesion_r:
        lesion = np.zeros_like(mask)
        cv2.circle(lesion, (150, 150), lesion_r, 1, -1)
        img[lesion > 0] = (25, 25, 25)
        mask[lesion > 0] = validate.LESION
    Image.fromarray(img).save(tmp_path / f"{name}.jpg")
    return mask


def test_sample_is_stratified_one_image_per_group(tmp_path):
    split = tmp_path / "split.csv"
    with open(split, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "split"])
        for c in validate.DISEASE_CLASSES:
            for i in range(30):
                w.writerow([f"{c}/{i}.jpg", c, f"{c}-{i // 2}", "test"])
            w.writerow([f"{c}/train.jpg", c, f"{c}-train", "train"])
    picked = validate.make_sample(60, split, tmp_path / "s.csv")
    assert len(picked) == 60
    assert {sum(p["label"] == c for p in picked) for c in validate.DISEASE_CLASSES} == {12}
    assert len({p["group"] for p in picked}) == 60 and all(p["split"] == "test" for p in picked)


def test_labelme_conversion_puts_lesions_on_top_of_leaf(tmp_path):
    shapes = [{"label": "lesion", "points": [[4, 4], [6, 4], [6, 6], [4, 6]]},
              {"label": "leaf", "points": [[1, 1], [9, 1], [9, 9], [1, 9]]}]
    j = tmp_path / "a.json"
    j.write_text(json.dumps({"imageHeight": 10, "imageWidth": 10, "shapes": shapes}))
    m = np.array(Image.open(validate.labelme_to_mask(j, tmp_path / "masks")))
    assert m[5, 5] == validate.LESION and m[2, 2] == validate.LEAF and m[0, 0] == validate.BACKGROUND


def test_evaluate_refuses_below_minimum_and_scores_perfect_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "REPO_ROOT", tmp_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    rows = []
    for i, r in enumerate([0, 20, 35, 45]):
        m = _leaf_image(tmp_path, f"im{i}", r)
        Image.fromarray(m).save(masks / f"im{i}.png")
        rows.append({"path": f"im{i}.jpg", "label": "Sooty Mold", "group": str(i), "mask_file": f"im{i}.png"})
    sample = tmp_path / "s.csv"
    with open(sample, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert validate.evaluate(masks, sample, tmp_path / "out", min_masks=50) is None
    res = validate.evaluate(masks, sample, tmp_path / "out", min_masks=4)
    assert res["overall"]["severity_mae_pp"]["mean"] < 2.0
    assert res["overall"]["leaf_iou"]["mean"] > 0.95
    assert res["overall"]["severity_spearman"] > 0.9
    assert (tmp_path / "out" / "per_image.csv").exists()
    assert "lesion_iou" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
