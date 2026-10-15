"""Clustering of change evidence and pairing of removed/added clusters into moves.

Added and removed evidence voxels are clustered separately with DBSCAN, and each cluster
gets a gravity-aligned box (yaw about +z only). A removed and an added cluster are paired
as a move when their sizes and colour histograms agree.

Box "extents" are full side lengths along (x', y', z); "yaw" rotates x' about +z, radians.

Reads  diff/evidence.npz, fusion/cloud_{A,B}.npz
Writes cluster/changes.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def gravity_obb(points: np.ndarray) -> dict[str, object]:
    """Yaw-only oriented bounding box of (N, 3) points.

    The yaw comes from PCA of the x-y coordinates; z bounds are axis-aligned.
    Returns center (3,), extents (3,), yaw (rad), corners (8, 3).
    """
    xy = points[:, :2]
    mu = xy.mean(axis=0)
    if len(points) >= 3:
        evals, evecs = np.linalg.eigh(np.cov((xy - mu).T))
        major = evecs[:, np.argmax(evals)]
    else:
        major = np.array([1.0, 0.0])
    yaw = float(np.arctan2(major[1], major[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    R2 = np.array([[c, -s], [s, c]])  # box axes in world (columns)
    local = (xy - mu) @ R2  # coordinates along box axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    zlo, zhi = points[:, 2].min(), points[:, 2].max()
    center_local = (lo + hi) / 2
    center_xy = mu + R2 @ center_local
    extents = np.array([hi[0] - lo[0], hi[1] - lo[1], zhi - zlo])
    corners = []
    for dx in (-0.5, 0.5):
        for dy in (-0.5, 0.5):
            for dz in (-0.5, 0.5):
                p = center_xy + R2 @ (np.array([dx * extents[0], dy * extents[1]]))
                corners.append([p[0], p[1], (zlo + zhi) / 2 + dz * extents[2]])
    return {
        "center": np.array([center_xy[0], center_xy[1], (zlo + zhi) / 2]),
        "extents": extents,
        "yaw": yaw,
        "corners": np.array(corners),
    }


def principal_extents(points: np.ndarray) -> np.ndarray:
    """Robust extents (2nd-98th percentile) along the 3 principal axes, ascending."""
    if len(points) < 4:
        return np.zeros(3)
    X = points - points.mean(axis=0)
    _, evecs = np.linalg.eigh(X.T @ X)
    proj = X @ evecs
    return np.sort(np.percentile(proj, 98, axis=0) - np.percentile(proj, 2, axis=0))


def planarity(points: np.ndarray) -> float:
    """Variance along the smallest principal axis / variance along the middle one: near 0
    for a flat patch, larger for a 3-D shell (an object seen from the front, side and
    top). 1.0 for fewer than 4 points."""
    if len(points) < 4:
        return 1.0
    X = points - points.mean(axis=0)
    w = np.linalg.eigvalsh(X.T @ X / len(X))
    return float(w[0] / max(w[1], 1e-12))


def cluster_points(
    points: np.ndarray,
    views: np.ndarray,
    cfg: dict,
    voxel_size: float,
    other_surface=None,
) -> list[dict]:
    """DBSCAN on evidence voxel centroids; drop small clusters and depth-bias artefacts.

    Monocular depth places a static surface slightly differently in the two recordings,
    which produces evidence with two typical shapes:
      - sheet: thin along one principal axis (< min_cluster_thickness) but extended along
        the middle one (> sheet_aspect * thickness). Slender objects such as poles or chair
        legs are thin along two axes and are kept.
      - patch: flat overall (`planarity` below `min_planarity`) even when depth noise
        makes it thicker than a sheet. An object's evidence is the shell the cameras saw
        (front, sides, top), which is not flat.
      - hugging: the evidence lies close to the other recording's surface (median distance
        to `other_surface`, a cKDTree of its fused cloud, < min_surface_gap). A removed
        object leaves the other recording looking at the background well behind it.

    Returns clusters with points, view support and box geometry. "occupied_volume" is
    (#evidence voxels) * voxel_size^3; "box_volume" is the OBB volume with each extent
    floored at one voxel (flat clusters would otherwise have zero volume).
    """
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN

    if len(points) == 0:
        return []
    # Evidence along an object's outline (where depth is unreliable) forms one-voxel-thin
    # lines that can bridge separate objects into one cluster; object evidence forms patches.
    if cfg.get("min_neighbors", 0) > 0:
        counts = np.array(
            [len(n) - 1 for n in cKDTree(points).query_ball_point(points, cfg["neighbor_radius"])]
        )
        keep = counts >= cfg["min_neighbors"]
        points, views = points[keep], views[keep]
    if len(points) == 0:
        return []
    labels = DBSCAN(eps=cfg["dbscan_eps"], min_samples=cfg["dbscan_min_points"]).fit_predict(points)
    clusters = []
    for lab in sorted(set(labels) - {-1}):
        m = labels == lab
        pts = points[m]
        if len(pts) < cfg["min_cluster_points"]:
            continue
        box = gravity_obb(pts)
        box_volume = float(np.prod(np.maximum(box["extents"], voxel_size)))
        ext = principal_extents(pts)
        # Size test on the two largest extents only: evidence covers the visible shell of an
        # object and is about one voxel thick, so a volume test would reject small objects.
        frontal_area = float(np.prod(np.maximum(ext[1:], voxel_size)))
        if frontal_area < cfg["min_frontal_area"]:
            continue
        if ext[0] < cfg["min_cluster_thickness"] and ext[1] > cfg["sheet_aspect"] * max(ext[0], voxel_size):
            continue  # sheet
        flatness = planarity(pts)
        if flatness < cfg["min_planarity"]:
            continue  # flat patch on a static surface
        gap = float(np.median(other_surface.query(pts)[0])) if other_surface is not None else None
        if gap is not None and gap < cfg["min_surface_gap"]:
            continue  # hugs the other epoch's surface
        clusters.append(
            {
                "points": pts,
                "num_voxels": int(len(pts)),
                "max_views": int(views[m].max()),
                "mean_views": float(views[m].mean()),
                "occupied_volume": float(len(pts) * voxel_size**3),
                "box_volume": box_volume,
                "frontal_area": frontal_area,
                "principal_extents": ext.tolist(),
                "planarity": flatness,
                "surface_gap": gap,
                **box,
            }
        )
    return clusters


def hue_sat_histogram(colors: np.ndarray, bins: int) -> np.ndarray | None:
    """Normalized 2-D hue/saturation histogram of RGB colors in [0, 1]. None if empty.

    Value (brightness) is ignored to reduce sensitivity to exposure changes.
    """
    if len(colors) == 0:
        return None
    import cv2

    rgb = np.clip(colors, 0, 1).astype(np.float32).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)  # H in [0,360), S in [0,1]
    hist, _, _ = np.histogram2d(hsv[:, 0], hsv[:, 1], bins=bins, range=[[0, 360], [0, 1]])
    return hist / hist.sum()


def colors_in_box(cloud_pts: np.ndarray, cloud_cols: np.ndarray, box: dict, margin: float) -> np.ndarray:
    """Colors of cloud points inside a (yaw-only) box, grown by `margin` on every side."""
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    R2 = np.array([[c, -s], [s, c]])
    local_xy = (cloud_pts[:, :2] - box["center"][:2]) @ R2
    half = box["extents"] / 2 + margin
    inside = (
        (np.abs(local_xy[:, 0]) <= half[0])
        & (np.abs(local_xy[:, 1]) <= half[1])
        & (np.abs(cloud_pts[:, 2] - box["center"][2]) <= half[2])
    )
    return cloud_cols[inside]


def pair_moved(removed: list[dict], added: list[dict], cfg: dict) -> list[dict]:
    """Optimal one-to-one REMOVED->ADDED pairing among geometrically compatible pairs.

    Compatible: box-volume ratio <= max_volume_ratio, every sorted-extent ratio
    <= max_extent_ratio, centroid distance <= max_centroid_distance, and appearance
    similarity (histogram intersection, if both histograms exist) >= min_appearance_sim.
    Cost = (1 - appearance) + normalized size mismatch.
    """
    if not removed or not added:
        return []
    INF = 1e9
    cost = np.full((len(removed), len(added)), INF)
    details = {}
    for i, r in enumerate(removed):
        for j, a in enumerate(added):
            vr = max(r["box_volume"], a["box_volume"]) / min(r["box_volume"], a["box_volume"])
            er = np.sort(np.maximum(r["extents"], 1e-9))
            ea = np.sort(np.maximum(a["extents"], 1e-9))
            ext_ratio = float(np.max(np.maximum(er / ea, ea / er)))
            dist = float(np.linalg.norm(a["center"] - r["center"]))
            if vr > cfg["max_volume_ratio"] or ext_ratio > cfg["max_extent_ratio"]:
                continue
            if dist > cfg["max_centroid_distance"]:
                continue
            hr, ha = r.get("hist"), a.get("hist")
            sim = float(np.minimum(hr, ha).sum()) if hr is not None and ha is not None else None
            if sim is not None and sim < cfg["min_appearance_sim"]:
                continue
            size_term = (np.log(vr) / np.log(cfg["max_volume_ratio"])) if cfg["max_volume_ratio"] > 1 else 0.0
            cost[i, j] = (1.0 - (sim if sim is not None else 0.5)) + size_term
            details[(i, j)] = {
                "volume_ratio": float(vr),
                "extent_ratio": ext_ratio,
                "appearance_similarity": sim,
            }
    rows, cols = linear_sum_assignment(cost)
    return [
        {"removed_idx": int(i), "added_idx": int(j), **details[(i, j)]}
        for i, j in zip(rows, cols)
        if cost[i, j] < INF
    ]


def _box_json(c: dict) -> dict:
    return {
        "center": np.round(c["center"], 4).tolist(),
        "extents": np.round(c["extents"], 4).tolist(),
        "yaw_rad": round(float(c["yaw"]), 4),
        "corners": np.round(c["corners"], 4).tolist(),
    }


def cluster_changes(
    out_dir: Path, cluster_cfg: dict, moved_cfg: dict, evidence_voxel_size: float
) -> list[dict]:
    """Stage entry: cluster evidence, pair MOVED, write cluster/changes.json."""
    ev = np.load(out_dir / "diff" / "evidence.npz")
    clouds = {e: np.load(out_dir / "fusion" / f"cloud_{e}.npz") for e in ("A", "B")}
    out = out_dir / "cluster"
    out.mkdir(parents=True, exist_ok=True)

    from scipy.spatial import cKDTree

    trees = {e: cKDTree(clouds[e]["points"]) for e in ("A", "B")}
    removed = cluster_points(
        ev["removed_points"], ev["removed_views"], cluster_cfg, evidence_voxel_size, trees["B"]
    )
    added = cluster_points(
        ev["added_points"], ev["added_views"], cluster_cfg, evidence_voxel_size, trees["A"]
    )
    margin = moved_cfg["appearance_box_margin"]
    for c in removed:  # removed object's appearance lives in epoch A
        c["hist"] = hue_sat_histogram(
            colors_in_box(clouds["A"]["points"], clouds["A"]["colors"], c, margin), moved_cfg["hist_bins"]
        )
    for c in added:  # added object's appearance lives in epoch B
        c["hist"] = hue_sat_histogram(
            colors_in_box(clouds["B"]["points"], clouds["B"]["colors"], c, margin), moved_cfg["hist_bins"]
        )
    pairs = pair_moved(removed, added, moved_cfg)
    paired_r = {p["removed_idx"] for p in pairs}
    paired_a = {p["added_idx"] for p in pairs}

    changes = []
    for p in pairs:
        r, a = removed[p["removed_idx"]], added[p["added_idx"]]
        disp = a["center"] - r["center"]
        changes.append(
            {
                "action": "moved",
                "box_before": _box_json(r),
                "box_after": _box_json(a),
                "displacement": np.round(disp, 4).tolist(),
                "distance": round(float(np.linalg.norm(disp)), 4),
                "support_views": int(min(r["max_views"], a["max_views"])),
                "num_voxels": int(r["num_voxels"] + a["num_voxels"]),
                "pairing": p,
            }
        )
    for i, c in enumerate(removed):
        if i not in paired_r:
            changes.append(
                {
                    "action": "removed",
                    "box": _box_json(c),
                    "support_views": c["max_views"],
                    "num_voxels": c["num_voxels"],
                }
            )
    for j, c in enumerate(added):
        if j not in paired_a:
            changes.append(
                {
                    "action": "added",
                    "box": _box_json(c),
                    "support_views": c["max_views"],
                    "num_voxels": c["num_voxels"],
                }
            )
    for k, ch in enumerate(changes):
        ch["id"] = k
    with open(out / "changes.json", "w") as f:
        json.dump(changes, f, indent=2)
    print(
        f"  Clusters: {len(removed)} removed, {len(added)} added -> "
        f"{sum(c['action'] == 'moved' for c in changes)} moved, "
        f"{sum(c['action'] == 'removed' for c in changes)} removed, "
        f"{sum(c['action'] == 'added' for c in changes)} added"
    )
    return changes
