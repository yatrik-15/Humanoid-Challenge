"""Visibility-aware change detection: render both grids from the same poses and compare.

For every pose of either recording, grids A and B are rendered from that pose. With
rendered z-depths dA, dB per pixel (0 = no hit):

    either render has no hit   -> skipped
    tau = max(tau_rel * max(dA, dB), tau_abs_min)
    added candidate    dB < dA - tau   evidence point = B's surface point
    removed candidate  dB > dA + tau   evidence point = A's surface point

A candidate only counts if the other recording observed that point as free space (voxel
weight > 0 and normalized TSDF > free_tsdf_min). Otherwise the other recording simply never
looked there (occlusion, missing coverage) and the point is recorded as unknown.

Evidence is accumulated on a coarse voxel grid; a voxel is kept when it is confirmed from
at least `min_views` distinct viewpoint groups (camera centres within
view_group_radius_rel * scene scale and viewing directions within view_group_angle_deg).

Conventions: src/utils/geometry.py. TSDF values are normalized to [-1, 1] (Open3D stores
tsdf / sdf_trunc) and positive in front of the surface.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.geometry import backproject_pixels, camera_center


class VoxelLookup:
    """Sparse voxel lookup: world point -> (weight, normalized tsdf). Unallocated -> weight 0.

    Built from flat arrays of integer voxel indices, so it is independent of the TSDF
    library. Voxel ijk holds the TSDF sampled at world position ijk * voxel_size (Open3D's
    voxel_coordinates / voxel_size is integral), so a query point maps to its nearest
    sample, round(X / voxel_size).
    """

    def __init__(self, voxel_ijk: np.ndarray, tsdf: np.ndarray, weight: np.ndarray, voxel_size: float):
        self.voxel_size = float(voxel_size)
        keys = self._encode(np.asarray(voxel_ijk, dtype=np.int64))
        order = np.argsort(keys)
        self.keys = keys[order]
        self.tsdf = np.asarray(tsdf, dtype=np.float32).ravel()[order]
        self.weight = np.asarray(weight, dtype=np.float32).ravel()[order]

    @staticmethod
    def _encode(ijk: np.ndarray) -> np.ndarray:
        """Pack (i, j, k) with |i|,|j|,|k| < 2^20 into one int64 key."""
        off = 1 << 20
        if np.any(np.abs(ijk) >= off):
            raise ValueError("voxel index out of encodable range")
        ijk = ijk + off
        return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]

    def query(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (weight, tsdf) at world points X (N, 3). Unallocated voxels: weight=0, tsdf=nan."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        w = np.zeros(len(X), dtype=np.float32)
        t = np.full(len(X), np.nan, dtype=np.float32)
        if len(X) == 0 or len(self.keys) == 0:
            return w, t
        k = self._encode(np.round(X / self.voxel_size).astype(np.int64))
        pos = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        hit = self.keys[pos] == k
        w[hit] = self.weight[pos[hit]]
        t[hit] = self.tsdf[pos[hit]]
        return w, t

    def observed_free(self, X: np.ndarray, free_tsdf_min: float) -> np.ndarray:
        """True where the grid observed X as empty space."""
        w, t = self.query(X)
        return (w > 0) & (t > free_tsdf_min)


def compare_renders(
    dA: np.ndarray,
    dB: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    look_A: VoxelLookup,
    look_B: VoxelLookup,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Per-pose comparison of two rendered z-depth maps from the same camera.

    Args:
        dA, dB: (H, W) z-depth renders of grid A / grid B (0 = no hit).
        K: 3x3 intrinsics of THIS render resolution (COLMAP convention).
        T_cw: world->camera pose of the render.
        look_A, look_B: voxel lookups used for the free-space confirmation.

    Returns:
        dict of (N, 3) world points: "added", "removed" (confirmed) and "unknown"
        (candidates whose free-space check failed).
    """
    both = (dA > 0) & (dB > 0)
    tau = np.maximum(cfg["tau_rel"] * np.maximum(dA, dB), cfg["tau_abs_min"])
    added_px = both & (dB < dA - tau)
    removed_px = both & (dB > dA + tau)

    def lift(mask: np.ndarray, d: np.ndarray) -> np.ndarray:
        rows, cols = np.nonzero(mask)
        return backproject_pixels(cols + 0.5, rows + 0.5, d[rows, cols], K, T_cw)

    X_add = lift(added_px, dB)
    X_rem = lift(removed_px, dA)
    if cfg.get("free_space_check", True):
        ok_add = look_A.observed_free(X_add, cfg["free_tsdf_min"])
        ok_rem = look_B.observed_free(X_rem, cfg["free_tsdf_min"])
    else:  # ablation only: trust every depth disagreement
        ok_add = np.ones(len(X_add), bool)
        ok_rem = np.ones(len(X_rem), bool)
    return {
        "added": X_add[ok_add],
        "removed": X_rem[ok_rem],
        "unknown": np.vstack([X_add[~ok_add], X_rem[~ok_rem]]),
    }


def view_groups(T_cws: list[np.ndarray], radius: float, angle_deg: float) -> np.ndarray:
    """Greedy grouping of poses into distinct viewpoints.

    A pose joins an existing group if its centre is within `radius` of the group's first
    pose AND its optical axis is within `angle_deg`; otherwise it starts a new group.
    Returns (N,) group ids.
    """
    cos_max = np.cos(np.deg2rad(angle_deg))
    reps: list[tuple[np.ndarray, np.ndarray]] = []
    ids = np.empty(len(T_cws), dtype=np.int64)
    for i, T in enumerate(T_cws):
        c, z = camera_center(T), T[2, :3]  # optical axis in world = third row of R_cw
        for g, (cg, zg) in enumerate(reps):
            if np.linalg.norm(c - cg) <= radius and float(np.dot(z, zg)) >= cos_max:
                ids[i] = g
                break
        else:
            reps.append((c, z))
            ids[i] = len(reps) - 1
    return ids


@dataclass
class EvidenceGrid:
    """Accumulates, per evidence voxel, the set of distinct view groups that confirmed it."""

    voxel_size: float
    views: dict[tuple[int, int, int], set] = field(default_factory=dict)
    points: dict[tuple[int, int, int], list[np.ndarray]] = field(default_factory=dict)

    def add(self, X: np.ndarray, group: int) -> None:
        if len(X) == 0:
            return
        ijk = np.floor(X / self.voxel_size).astype(np.int64)
        uniq, inv = np.unique(ijk, axis=0, return_inverse=True)
        inv = inv.ravel()
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inv, X)
        counts = np.bincount(inv, minlength=len(uniq))
        for key, s, n in zip(map(tuple, uniq), sums, counts):
            self.views.setdefault(key, set()).add(group)
            self.points.setdefault(key, []).append(s / n)

    def confirmed(self, min_views: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (M, 3) voxel centroids (mean of evidence points) and (M,) view counts."""
        keys = [k for k, v in self.views.items() if len(v) >= min_views]
        if not keys:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
        cents = np.array([np.mean(self.points[k], axis=0) for k in keys])
        counts = np.array([len(self.views[k]) for k in keys])
        return cents, counts


def accumulate_changes(
    renders: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    look_A: VoxelLookup,
    look_B: VoxelLookup,
    cfg: dict,
    scene_scale: float,
) -> dict[str, dict]:
    """Run compare_renders over all poses and keep multi-view-confirmed evidence voxels.

    Args:
        renders: list of (dA, dB, K, T_cw), one per pose in the union of A and B poses.
        scene_scale: length used to make view_group_radius scale-relative.

    Returns:
        {"added"|"removed"|"unknown": {"points": (M,3), "views": (M,)}}. "unknown" keeps
        every voxel with any unknown evidence (views >= 1), for visualization only.
    """
    groups = view_groups(
        [r[3] for r in renders], cfg["view_group_radius_rel"] * scene_scale, cfg["view_group_angle_deg"]
    )
    grids = {k: EvidenceGrid(cfg["evidence_voxel_size"]) for k in ("added", "removed", "unknown")}
    for (dA, dB, K, T_cw), g in zip(renders, groups):
        res = compare_renders(dA, dB, K, T_cw, look_A, look_B, cfg)
        for k in grids:
            grids[k].add(res[k], int(g))
    out = {}
    for k, grid in grids.items():
        pts, views = grid.confirmed(cfg["min_views"] if k != "unknown" else 1)
        out[k] = {"points": pts, "views": views}
    out["num_view_groups"] = int(groups.max() + 1) if len(groups) else 0
    return out


def detect_changes(out_dir: Path, cfg: dict, fusion_cfg: dict) -> dict[str, object]:
    """Stage entry: render both grids from the union of A and B poses and accumulate evidence.

    Inputs:  align/poses.json, fusion/grid_{A,B}.npz
    Outputs: diff/evidence.npz (added/removed/unknown points + view counts),
             diff/diff_report.json
    """
    import open3d as o3d

    from src.fusion import DepthRenderer, grid_lookup

    poses = json.load(open(out_dir / "align" / "poses.json"))
    fdir = out_dir / "fusion"
    grids = {e: o3d.t.geometry.VoxelBlockGrid.load(str(fdir / f"grid_{e}.npz")) for e in ("A", "B")}
    looks = {e: grid_lookup(g, fusion_cfg["voxel_size"]) for e, g in grids.items()}
    renderers = {e: DepthRenderer(g, fusion_cfg) for e, g in grids.items()}

    s = cfg["render_scale"]
    renders = []
    for name in sorted(poses):
        p = poses[name]
        W, H = int(round(p["width"] * s)), int(round(p["height"] * s))
        K = np.array(p["K"], dtype=np.float64)
        K[:2] *= s  # exact for COLMAP corner-origin image coordinates
        T = np.array(p["T_cw"], dtype=np.float64)
        renders.append((renderers["A"].render(K, T, W, H), renderers["B"].render(K, T, W, H), K, T))

    cams = np.array([camera_center(np.array(p["T_cw"])) for p in poses.values()])
    cloud = np.load(fdir / "cloud_A.npz")["points"]
    scale = float(np.median(np.linalg.norm(cloud - cams.mean(axis=0), axis=1))) if len(cloud) else 1.0
    ev = accumulate_changes(renders, looks["A"], looks["B"], cfg, scale)

    ddir = out_dir / "diff"
    ddir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ddir / "evidence.npz",
        added_points=ev["added"]["points"],
        added_views=ev["added"]["views"],
        removed_points=ev["removed"]["points"],
        removed_views=ev["removed"]["views"],
        unknown_points=ev["unknown"]["points"],
        unknown_views=ev["unknown"]["views"],
    )
    report = {
        "num_poses_rendered": len(renders),
        "num_view_groups": ev["num_view_groups"],
        "render_scale": s,
        "scene_scale": scale,
        "confirmed_added_voxels": int(len(ev["added"]["points"])),
        "confirmed_removed_voxels": int(len(ev["removed"]["points"])),
        "unknown_voxels": int(len(ev["unknown"]["points"])),
        "free_space_check": bool(cfg.get("free_space_check", True)),
    }
    with open(ddir / "diff_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"  Rendered {len(renders)} poses ({ev['num_view_groups']} view groups); confirmed voxels: "
        f"{report['confirmed_added_voxels']} added, {report['confirmed_removed_voxels']} removed, "
        f"{report['unknown_voxels']} unknown"
    )
    return report
