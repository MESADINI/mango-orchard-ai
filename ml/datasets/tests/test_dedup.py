import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ml.datasets.dedup import (
    candidate_pairs, geometric_inliers, group_near_duplicates, phash_bits, phash_min_distance, sift_features,
    union_find_groups,
)


def _leaf(seed: int, size=(240, 320)) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, (230, 235, 240))
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x, y = rng.integers(0, size[0] - 40), rng.integers(0, size[1] - 40)
        w, h = rng.integers(20, 120, size=2)
        draw.ellipse([x, y, x + w, y + h], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    return img


def test_union_find_groups_connected_components():
    groups = union_find_groups(6, np.array([[0, 1], [1, 2], [4, 5]]))
    assert groups[0] == groups[1] == groups[2]
    assert groups[4] == groups[5]
    assert len({groups[0], groups[3], groups[4]}) == 3


def test_rotated_and_mirrored_copies_have_zero_phash_distance(tmp_path):
    a = _leaf(1)
    paths = []
    for name, img in [("a", a), ("a_rot90", a.rotate(90, expand=True)), ("a_mirror", ImageOps.mirror(a)),
                      ("b", _leaf(2))]:
        p = tmp_path / f"{name}.png"
        img.save(p)
        paths.append(p)
    dist = phash_min_distance(phash_bits(paths))
    assert dist[0, 1] == 0 and dist[0, 2] == 0
    assert dist[0, 3] > 0
    assert (dist == dist.T).all()


def test_group_near_duplicates_combines_both_signals():
    phash = np.full((4, 4), 30, dtype=np.int16)
    phash[0, 1] = phash[1, 0] = 0                      # pHash link 0-1
    pairs = np.array([[2, 3], [1, 2]])
    inliers = np.array([40, 5])                        # verified link 2-3; 1-2 below threshold
    out = group_near_duplicates(phash, pairs, inliers, phash_threshold=0, inlier_threshold=25)
    g = out["groups"]
    assert g[0] == g[1] and g[2] == g[3] and g[0] != g[2]
    assert out["n_phash_edges"] == 1 and out["n_verified_edges"] == 1


def test_candidate_pairs_are_unique_sorted_and_exclude_self():
    cos = np.array([[-1, .9, .1], [.9, -1, .2], [.1, .2, -1]], dtype=np.float32)
    pairs = candidate_pairs(cos, 1)
    assert pairs.tolist() == [[0, 1], [1, 2]]          # 0->1, 1->0, 2->1
    assert (pairs[:, 0] < pairs[:, 1]).all()


def _textured_leaf(seed: int, size=(320, 240)) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(img)
    for _ in range(150):
        x, y = rng.integers(0, size[0]), rng.integers(0, size[1])
        r = int(rng.integers(2, 9))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(c) for c in rng.integers(0, 200, 3)))
    return img


def test_geometric_verification_links_zoomed_rotated_mirrored_copy_but_not_other_leaf(tmp_path):
    base = _textured_leaf(1)
    zoomed = base.crop((40, 30, 280, 210)).resize((320, 240)).rotate(90, expand=True)
    mirrored = ImageOps.mirror(base)
    other = _textured_leaf(2)
    paths = []
    for name, im in (("base", base), ("zoom", zoomed), ("mirror", mirrored), ("other", other)):
        paths.append(tmp_path / f"{name}.png")
        im.save(paths[-1])
    feats = sift_features(paths)
    assert geometric_inliers(feats[0], feats[1]) >= 25
    assert geometric_inliers(feats[0], feats[2]) >= 25
    assert geometric_inliers(feats[0], feats[3]) < 25


def test_threshold_sweep_counts_cross_class_links():
    from ml.datasets.build_disease_dataset import threshold_sweep

    ph = np.full((4, 4), 64, dtype=np.int16)
    pairs = np.array([[0, 1], [0, 2], [2, 3]])
    inliers = np.array([50, 20, 10])                   # 0-1 same class; 0-2 cross class; 2-3 same class
    loose, strict = threshold_sweep(ph, pairs, inliers, ["A", "A", "B", "B"], 0, [10, 30])
    assert loose["n_groups"] == 1 and loose["n_cross_class_links"] == 1
    assert loose["n_cross_class_conflict_groups"] == 1 and loose["n_images_in_conflict_groups"] == 4
    assert loose["false_link_rate_estimate"] == 1.0
    assert strict["n_groups"] == 3 and strict["n_cross_class_links"] == 0 and strict["largest_group"] == 2


def test_oversized_groups_always_train_and_never_held_out():
    from ml.datasets.build_disease_dataset import ALWAYS_TRAIN, stratified_group_split

    rng = np.random.default_rng(0)
    labels = np.repeat(["A", "B"], 400)
    groups = np.arange(800)
    groups[:60] = 0                                    # one class-A leaf copied 60 times (> 5% of 400)
    split, cv = stratified_group_split(labels, groups)
    assert set(split[:60]) == {"train"} and set(cv[:60]) == {ALWAYS_TRAIN}
    assert set(cv[split == "test"]) == {-1}
    assert set(cv[(split != "test") & (groups != 0)]) == {0, 1, 2, 3, 4}
    for s, target in (("val", 0.15), ("test", 0.15)):
        assert abs((split[labels == "B"] == s).mean() - target) < 0.03


def test_geometric_inliers_never_exceed_keypoints_of_either_image(tmp_path):
    # a richly textured image vs a nearly blank one: many-to-one matches must not inflate the inlier count
    rich = _textured_leaf(3, size=(1000, 1000))
    blank = Image.new("RGB", (320, 240), (235, 235, 235))
    ImageDraw.Draw(blank).ellipse([100, 80, 140, 120], fill=(30, 120, 30))
    paths = [tmp_path / "rich.png", tmp_path / "blank.png"]
    rich.save(paths[0]); blank.save(paths[1])
    feats = sift_features(paths)
    n_blank = len(feats[1][0][0])
    assert geometric_inliers(feats[0], feats[1]) <= n_blank
    assert geometric_inliers(feats[1], feats[0]) <= n_blank
