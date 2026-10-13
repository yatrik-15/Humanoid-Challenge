"""Naive baseline for the ablation: nearest-neighbour distance between the fused clouds.

A point of A farther than `nn_threshold` from every point of B counts as removed, and vice
versa for added. There is no visibility reasoning, so anything one recording did not cover
looks like a change. Clustering and move pairing reuse the main method's code and settings.

Reads fusion/cloud_{A,B}.npz; writes baseline/changes.json (same schema as cluster/).
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.cluster import _box_json, cluster_points, colors_in_box, hue_sat_histogram, pair_moved


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Mean point per voxel (keeps the baseline's density comparable to evidence voxels)."""
    if len(points) == 0:
        return points
    ijk = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(ijk, axis=0, return_inverse=True)
    inv = inv.ravel()
    sums = np.zeros((inv.max() + 1, 3))
    np.add.at(sums, inv, points)
    return sums / np.bincount(inv)[:, None]


def naive_changes(
    out_dir: Path, baseline_cfg: dict, cluster_cfg: dict, moved_cfg: dict, voxel: float
) -> list[dict]:
    clouds = {e: np.load(out_dir / "fusion" / f"cloud_{e}.npz") for e in ("A", "B")}
    A, B = clouds["A"]["points"], clouds["B"]["points"]
    thr = baseline_cfg["nn_threshold"]
    tA, tB = cKDTree(A), cKDTree(B)
    rem = voxel_downsample(A[tB.query(A)[0] > thr], voxel)
    add = voxel_downsample(B[tA.query(B)[0] > thr], voxel)
    # same clustering AND artefact filters as the main method, so the ablation isolates
    # the visibility reasoning (free-space check, multi-view confirmation)
    removed = cluster_points(rem, np.zeros(len(rem), int), cluster_cfg, voxel, tB)
    added = cluster_points(add, np.zeros(len(add), int), cluster_cfg, voxel, tA)
    m = moved_cfg["appearance_box_margin"]
    for c in removed:
        c["hist"] = hue_sat_histogram(colors_in_box(A, clouds["A"]["colors"], c, m), moved_cfg["hist_bins"])
    for c in added:
        c["hist"] = hue_sat_histogram(colors_in_box(B, clouds["B"]["colors"], c, m), moved_cfg["hist_bins"])
    pairs = pair_moved(removed, added, moved_cfg)
    pr, pa = {p["removed_idx"] for p in pairs}, {p["added_idx"] for p in pairs}
    out = []
    for p in pairs:
        r, a = removed[p["removed_idx"]], added[p["added_idx"]]
        disp = a["center"] - r["center"]
        out.append(
            {
                "action": "moved",
                "box_before": _box_json(r),
                "box_after": _box_json(a),
                "displacement": np.round(disp, 4).tolist(),
                "distance": round(float(np.linalg.norm(disp)), 4),
            }
        )
    out += [{"action": "removed", "box": _box_json(c)} for i, c in enumerate(removed) if i not in pr]
    out += [{"action": "added", "box": _box_json(c)} for j, c in enumerate(added) if j not in pa]
    for k, ch in enumerate(out):
        ch["id"] = k
    bdir = out_dir / "baseline"
    bdir.mkdir(parents=True, exist_ok=True)
    with open(bdir / "changes.json", "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"  Naive NN baseline ({thr}): "
        + ", ".join(f"{sum(c['action'] == a for c in out)} {a}" for a in ("moved", "removed", "added"))
    )
    return out
