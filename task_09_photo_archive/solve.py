#!/usr/bin/env python3
"""
Task 9: Photo Archive
Instance-level grouping of masked image crops with hard cannot-link constraints.

Input:
    fragment_album/
      scene_0001/
        crops.csv
        distances.csv
        crops/*.jpg
        masks/*.png
      ...
Output:
    submission.csv

Dependencies:
    numpy
    opencv-python (SIFT support)
    Pillow

The distance table is interpreted as a hard cannot-link relation:
two crops connected by a saved distance must never receive the same object_id.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


TOP_K = 60
WEAK_MIN_SIM = 0.55
WEAK_MIN_INLIERS = 3
STRONG_MIN_INLIERS = 5
MIN_MUTUAL_MATCHES = 3


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_global_descriptor(
    image: np.ndarray,
    mask: np.ndarray,
    hog_descriptor: cv2.HOGDescriptor,
) -> np.ndarray:
    h, w = mask.shape
    scale = 64.0 / max(h, w)
    if scale > 2.5:
        scale = 2.5

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
    resized_mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((64, 64, 3), 127, dtype=np.uint8)
    mask_canvas = np.zeros((64, 64), dtype=np.uint8)

    ox = (64 - nw) // 2
    oy = (64 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    mask_canvas[oy:oy + nh, ox:ox + nw] = resized_mask
    canvas[mask_canvas == 0] = 127

    hsv = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], mask_canvas,
        [12, 4, 4], [0, 180, 0, 256, 0, 256]
    ).flatten().astype(np.float32)
    hist /= float(hist.sum()) + 1e-8
    hist /= float(np.linalg.norm(hist)) + 1e-8

    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    hog = hog_descriptor.compute(gray).flatten().astype(np.float32)
    hog /= float(np.linalg.norm(hog)) + 1e-8

    silhouette = cv2.resize(
        mask_canvas, (16, 16), interpolation=cv2.INTER_AREA
    ).astype(np.float32).flatten() / 255.0
    silhouette /= float(np.linalg.norm(silhouette)) + 1e-8

    descriptor = np.concatenate(
        (0.30 * hist, 0.50 * hog, 0.20 * silhouette)
    ).astype(np.float32)
    descriptor /= float(np.linalg.norm(descriptor)) + 1e-8
    return descriptor


def extract_scene_features(
    scene_dir: Path,
    crop_rows: List[dict],
    hog_descriptor: cv2.HOGDescriptor,
    sift_detector: cv2.SIFT,
):
    global_features = []
    local_features = []

    for row in crop_rows:
        image = np.asarray(
            Image.open(scene_dir / row["image_path"]).convert("RGB")
        )
        mask = np.asarray(
            Image.open(scene_dir / row["mask_path"]).convert("L")
        )

        global_features.append(
            make_global_descriptor(image, mask, hog_descriptor)
        )

        h, w = mask.shape
        scale = 160.0 / max(h, w)
        if scale > 5.0:
            scale = 5.0
        if scale < 0.35:
            scale = 0.35

        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

        image_s = cv2.resize(image, (nw, nh), interpolation=interpolation)
        mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(image_s, cv2.COLOR_RGB2GRAY)
        gray[mask_s == 0] = 127

        keypoints, descriptors = sift_detector.detectAndCompute(gray, mask_s)
        local_features.append((keypoints, descriptors))

    return np.vstack(global_features), local_features


def sift_pair(
    keypoints_a,
    descriptors_a,
    keypoints_b,
    descriptors_b,
) -> Tuple[int, int, int]:
    if (
        descriptors_a is None or descriptors_b is None
        or len(descriptors_a) < 2 or len(descriptors_b) < 2
    ):
        return 0, 0, 0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    try:
        matches_ab = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        matches_ba = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    except cv2.error:
        return 0, 0, min(len(keypoints_a), len(keypoints_b))

    good_ab = [
        m for m, n in matches_ab
        if m.distance < 0.75 * n.distance
    ]
    good_ba = [
        m for m, n in matches_ba
        if m.distance < 0.75 * n.distance
    ]

    map_ab = {m.queryIdx: m.trainIdx for m in good_ab}
    map_ba = {m.queryIdx: m.trainIdx for m in good_ba}
    mutual = [
        (i, j) for i, j in map_ab.items()
        if map_ba.get(j) == i
    ]

    inliers = 0
    if len(mutual) >= 3:
        src = np.float32(
            [keypoints_a[i].pt for i, _ in mutual]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [keypoints_b[j].pt for _, j in mutual]
        ).reshape(-1, 1, 2)

        try:
            _, inlier_mask = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=1000,
                confidence=0.99,
                refineIters=10,
            )
            if inlier_mask is not None:
                inliers = int(inlier_mask.sum())
        except cv2.error:
            pass

    min_keypoints = min(len(keypoints_a), len(keypoints_b))
    return len(mutual), inliers, min_keypoints


def cluster_scene(
    scene_dir: Path,
    top_k: int = TOP_K,
) -> Tuple[List[Tuple[str, str, int]], dict]:
    crop_rows = read_csv(scene_dir / "crops.csv")
    distance_rows = read_csv(scene_dir / "distances.csv")

    crop_ids = [row["crop_id"] for row in crop_rows]
    n = len(crop_ids)
    index = {crop_id: i for i, crop_id in enumerate(crop_ids)}

    hog_descriptor = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    sift_detector = cv2.SIFT_create(
        nfeatures=180,
        contrastThreshold=0.012,
        edgeThreshold=10,
    )

    global_features, local_features = extract_scene_features(
        scene_dir, crop_rows, hog_descriptor, sift_detector
    )
    similarity = global_features @ global_features.T

    cannot_link = set()
    for row in distance_rows:
        a = index[row["crop_id_a"]]
        b = index[row["crop_id_b"]]
        if a > b:
            a, b = b, a
        cannot_link.add((a, b))

    # Candidate pairs: visual nearest neighbours plus every known
    # cannot-link pair (needed for the final hard-constraint check).
    candidates = set(cannot_link)
    for i in range(n):
        if n <= top_k + 1:
            neighbours = np.argsort(-similarity[i])[1:]
        else:
            neighbours = np.argpartition(
                -similarity[i], top_k + 1
            )[:top_k + 1]
            neighbours = neighbours[neighbours != i]

        for j in neighbours:
            j = int(j)
            if i != j:
                if i < j:
                    candidates.add((i, j))
                else:
                    candidates.add((j, i))

    pair_data = {}
    for i, j in candidates:
        pair_data[(i, j)] = sift_pair(
            local_features[i][0],
            local_features[i][1],
            local_features[j][0],
            local_features[j][1],
        )

    # Disjoint-set union with explicit cannot-link sets.
    parent = list(range(n))
    members = [{i} for i in range(n)]
    forbidden = [set() for _ in range(n)]

    for i, j in cannot_link:
        forbidden[i].add(j)
        forbidden[j].add(i)

    edges = []
    for (i, j), (mutual, inliers, _min_kp) in pair_data.items():
        global_sim = float(similarity[i, j])

        is_match = (
            mutual >= MIN_MUTUAL_MATCHES
            and (
                inliers >= STRONG_MIN_INLIERS
                or (
                    inliers >= WEAK_MIN_INLIERS
                    and global_sim >= WEAK_MIN_SIM
                )
            )
        )

        if is_match and (i, j) not in cannot_link:
            edges.append((inliers, mutual, global_sim, i, j))

    # Deterministic descending order; index tie-breakers make the
    # result stable when visual scores are equal.
    edges.sort(reverse=True)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merge_count = 0
    for _inliers, _mutual, _sim, i, j in edges:
        root_i = find(i)
        root_j = find(j)
        if root_i == root_j:
            continue

        if forbidden[root_i].intersection(members[root_j]):
            continue
        if forbidden[root_j].intersection(members[root_i]):
            continue

        if len(members[root_i]) < len(members[root_j]):
            root_i, root_j = root_j, root_i

        parent[root_j] = root_i
        members[root_i].update(members[root_j])
        forbidden[root_i].update(forbidden[root_j])
        members[root_j] = set()
        forbidden[root_j] = set()
        merge_count += 1

    groups = {}
    for i, crop_id in enumerate(crop_ids):
        groups.setdefault(find(i), []).append(crop_id)

    groups = sorted(groups.values(), key=lambda group: min(group))

    object_id = {}
    for group_id, group in enumerate(groups):
        for crop_id in group:
            object_id[crop_id] = group_id

    # Hard QC: no known different-object pair may be merged.
    violations = []
    for i, j in cannot_link:
        if object_id[crop_ids[i]] == object_id[crop_ids[j]]:
            violations.append((crop_ids[i], crop_ids[j]))

    if violations:
        raise RuntimeError(
            f"{scene_dir.name}: cannot-link violations: {violations[:5]}"
        )

    output_rows = [
        (scene_dir.name, crop_id, object_id[crop_id])
        for crop_id in crop_ids
    ]

    stats = {
        "scene": scene_dir.name,
        "crops": n,
        "known_different_pairs": len(cannot_link),
        "candidate_pairs": len(candidates),
        "positive_match_edges": len(edges),
        "clusters": len(groups),
        "singletons": sum(len(group) == 1 for group in groups),
        "largest_cluster": max(map(len, groups)),
        "accepted_merges": merge_count,
        "cannot_link_violations": len(violations),
    }
    return output_rows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the extracted fragment_album directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
    )
    args = parser.parse_args()

    scene_dirs = sorted(
        path for path in args.data.iterdir()
        if path.is_dir() and path.name.startswith("scene_")
    )

    if not scene_dirs:
        raise SystemExit(f"No scene_* directories found in {args.data}")

    all_rows = []
    all_stats = []

    for scene_dir in scene_dirs:
        rows, stats = cluster_scene(scene_dir)
        all_rows.extend(rows)
        all_stats.append(stats)
        print(
            f'{stats["scene"]}: crops={stats["crops"]}, '
            f'clusters={stats["clusters"]}, '
            f'singletons={stats["singletons"]}, '
            f'cannot_link_violations={stats["cannot_link_violations"]}'
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["scene_id", "crop_id", "object_id"])
        writer.writerows(all_rows)

    total_crops = sum(s["crops"] for s in all_stats)
    total_clusters = sum(s["clusters"] for s in all_stats)
    total_violations = sum(s["cannot_link_violations"] for s in all_stats)

    print(f"Scenes: {len(all_stats)}")
    print(f"Crops: {total_crops}")
    print(f"Predicted clusters: {total_clusters}")
    print(f"Cannot-link violations: {total_violations}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
