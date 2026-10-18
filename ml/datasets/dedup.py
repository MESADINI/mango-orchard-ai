"""
Near-duplicate detection and grouping for image datasets.

MangoLeafBD's authors state that only ~1,800 of its 4,000 images are distinct leaves; the rest were made by
zooming and rotating. A random split would put copies of the same leaf in train and test (leakage).

Two link types, both invariant to rotation and mirroring:
  1. Perceptual hash (pHash) Hamming distance over the 8 dihedral views - catches re-encoded / rotated copies.
     On this data pHash is only trustworthy at distance 0 (different leaves collide at distance >= 4).
  2. Retrieval + geometric verification for zoomed / cropped copies: the k most similar images by DINOv2
     (dihedral max cosine) are candidates, and a candidate pair is linked only if SIFT matches agree on one
     similarity transform (RANSAC inliers >= threshold). DINOv2 similarity alone is NOT used as a link: it cannot
     separate "same leaf" from "another leaf of the same class" (visually verified, docs/phases/PHASE_1.md).
Linked images are merged with union-find; each connected component is one "group" (assumed to be one physical
leaf). Splits are then made by group.
"""

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageOps

DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"


def dihedral_views(img: Image.Image) -> List[Image.Image]:
    views = []
    for base in (img, ImageOps.mirror(img)):
        for k in range(4):
            views.append(base.rotate(90 * k, expand=True))
    return views


def phash_bits(paths: Sequence[Path]) -> np.ndarray:
    """Returns uint8 array [N, 8, 64]: pHash bits of each dihedral view."""
    import imagehash

    out = np.zeros((len(paths), 8, 64), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            views = dihedral_views(im.convert("RGB"))
        out[i] = [imagehash.phash(v).hash.flatten() for v in views]
    return out


def phash_min_distance(bits: np.ndarray) -> np.ndarray:
    """[N, N] minimum Hamming distance between image i (as-is) and any dihedral view of image j."""
    base = bits[:, 0, :]
    dist = np.full((len(bits), len(bits)), 64, dtype=np.int16)
    for t in range(bits.shape[1]):
        dist = np.minimum(dist, (base[:, None, :] != bits[None, :, t, :]).sum(-1))
    dist = np.minimum(dist, dist.T)
    np.fill_diagonal(dist, 64)
    return dist


def embedding_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def dino_embeddings(paths: Sequence[Path], batch_images: int = 16, device: str = "") -> np.ndarray:
    """Returns float32 [N, 8, D] L2-normalised DINOv2 embeddings of each dihedral view (fp32, GPU if available)."""
    import timm
    import torch

    device = device or embedding_device()
    model = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0, img_size=224).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg["input_size"] = (3, 224, 224)
    tf = timm.data.create_transform(**cfg)
    out = np.zeros((len(paths), 8, model.num_features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(paths), batch_images):
            views = []
            for p in paths[start:start + batch_images]:
                with Image.open(p) as im:
                    views += [tf(v) for v in dihedral_views(im.convert("RGB"))]
            feats = torch.nn.functional.normalize(model(torch.stack(views).to(device)), dim=-1)
            out[start:start + len(views) // 8] = feats.cpu().numpy().reshape(-1, 8, feats.shape[-1])
    return out


def dihedral_max_cosine(emb: np.ndarray) -> np.ndarray:
    """[N, N] max cosine similarity between image i (as-is) and any dihedral view of image j."""
    base = emb[:, 0, :]
    sim = np.full((len(emb), len(emb)), -1.0, dtype=np.float32)
    for t in range(emb.shape[1]):
        sim = np.maximum(sim, base @ emb[:, t, :].T)
    sim = np.maximum(sim, sim.T)
    np.fill_diagonal(sim, -1.0)
    return sim


def union_find_groups(n: int, edges: np.ndarray) -> np.ndarray:
    """Connected components of an undirected graph given as an [E, 2] edge array; returns group id per node."""
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def group_near_duplicates(phash_dist: np.ndarray, pairs: np.ndarray, inliers: np.ndarray,
                          phash_threshold: int, inlier_threshold: int) -> Dict[str, object]:
    """Union-find over (pHash distance <= phash_threshold) and (candidate pair with >= inlier_threshold inliers)."""
    pi, pj = np.where(np.triu(phash_dist <= phash_threshold, 1))
    verified = pairs[inliers >= inlier_threshold].reshape(-1, 2)
    edges = np.concatenate([np.stack([pi, pj], axis=1), verified]).astype(int)
    groups = union_find_groups(len(phash_dist), edges)
    return {"groups": groups, "n_phash_edges": int(len(pi)), "n_verified_edges": int(len(verified))}


# ----------------------------------------------------------------------------- geometric verification
# Global embeddings cannot tell "the same physical leaf, zoomed/rotated" from "another healthy leaf on the same white
# background" (visually checked: many same-class pairs at cosine >= 0.96 are different leaves). Copies made by
# zooming/rotating/mirroring preserve the local geometry of veins and lesions, so a pair is accepted as the same leaf
# only if SIFT matches agree on one similarity transform (rotation + uniform scale + translation, optionally mirrored).
SIFT_MAX_FEATURES = 1000
RATIO_TEST = 0.75


SIFT_MAX_SIDE = 640  # larger images (external camera photos up to 4000 px) are downscaled; MangoLeafBD is 320 px


def sift_features(paths: Sequence[Path], max_features: int = SIFT_MAX_FEATURES, max_side: int = SIFT_MAX_SIDE):
    """Per image: (keypoints [K,2] float32, descriptors [K,128] float32) for the image and for its mirror."""
    import cv2

    sift = cv2.SIFT_create(nfeatures=max_features)
    feats = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
        pair = []
        for img in (gray, gray[:, ::-1].copy()):
            kps, desc = sift.detectAndCompute(img, None)
            pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
            pair.append((pts, desc if desc is not None else np.zeros((0, 128), np.float32)))
        feats.append(pair)
    return feats


def geometric_inliers(fa, fb, ratio: float = RATIO_TEST) -> int:
    """Max RANSAC inliers of a similarity transform between image a and (image b or its mirror)."""
    import cv2

    pa, da = fa[0]
    best = 0
    if len(da) < 3:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for pb, db in fb:
        if len(db) < 3:
            continue
        knn = matcher.knnMatch(da, db, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
        # one-to-one: keep only the best match per keypoint of b. Without this, a high-resolution image with 1000
        # keypoints "matches" a low-texture image with 20 keypoints hundreds of times (inliers > keypoints).
        best_for_train = {}
        for m in good:
            if m.trainIdx not in best_for_train or m.distance < best_for_train[m.trainIdx].distance:
                best_for_train[m.trainIdx] = m
        good = list(best_for_train.values())
        if len(good) < 3:
            continue
        src = pa[[m.queryIdx for m in good]]
        dst = pb[[m.trainIdx for m in good]]
        _, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                              maxIters=2000, confidence=0.99)
        if mask is not None:
            best = max(best, int(mask.sum()))
    return best


def candidate_pairs(cosine: np.ndarray, k: int) -> np.ndarray:
    """[P, 2] unique pairs (i < j) where j is among i's k most similar images (or vice versa)."""
    nn = np.argsort(-cosine, axis=1)[:, :k]
    i = np.repeat(np.arange(len(cosine)), k)
    pairs = np.sort(np.stack([i, nn.ravel()], axis=1), axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0)


def verify_pairs(feats, pairs: np.ndarray) -> np.ndarray:
    """Inlier count for each candidate pair."""
    return np.array([geometric_inliers(feats[a], feats[b]) for a, b in pairs], dtype=np.int32)
